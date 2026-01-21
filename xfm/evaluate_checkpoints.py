import torch
import re
from pathlib import Path
from .config import build_rf
from .dataset.build_dataset import build_dataloaders
from .utils import set_seed, load_checkpoint
from .evaluation.evaluation_utils import get_fid_components, get_real_features_for_dataset, get_vae, evaluate
from torch.utils.tensorboard import SummaryWriter
import hydra
from omegaconf import DictConfig, OmegaConf

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.allow_tf32 = True


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    if "run_dir" not in cfg or cfg.run_dir is None:
        return
    run_dir = Path(cfg.run_dir)
    saved_config_path = run_dir / ".hydra" / "config.yaml"

    if not saved_config_path.exists():
        print(f"No config found at {saved_config_path}.")
        return
    
    print(f"Loading config from {saved_config_path}")
    saved_cfg = OmegaConf.load(saved_config_path)
    eval_overrides = cfg.get("evaluation", {})
    cfg = OmegaConf.merge(saved_cfg, {"evaluation": eval_overrides})
    set_seed(cfg.seed)

    eval_output_dir = run_dir / "evaluation"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    logger = SummaryWriter(log_dir=eval_output_dir)

    rf = build_rf(cfg)
    train_loader, test_loader = build_dataloaders(cfg)

    loader = test_loader if cfg.evaluation.use_test_set else train_loader

    vae = get_vae(cfg)
    fid_model, fid_resizer, fid_stats = get_fid_components(cfg.dataset.name, cfg.device)
    if vae:
        vae.to(cfg.device)
    
    print("Collecting real features.")
    real_feats = get_real_features_for_dataset(loader, fid_model, fid_resizer, cfg.device, max_batches=cfg.evaluation.eval_batches, vae=vae)
    if vae:
        vae.to("cpu")

    checkpoint_paths = sorted(
        run_dir.glob("*.pt"),
        key=lambda p: int(re.search(r'\d+', p.name).group())
    )

    if not checkpoint_paths:
        print(f"No .pt files found in {run_dir}")
        return

    if cfg.evaluation.only_final:
        checkpoint_paths = checkpoint_paths[-1:]

    print(f"Found {len(checkpoint_paths)} checkpoint(s) to evaluate in {run_dir}")

    for i, checkpoint_path in enumerate(checkpoint_paths):
        print(f"Processing {checkpoint_path.name}")
        global_step = load_checkpoint(rf.model, None, None, None, checkpoint_path, load_ema_weights=True)
        
        for cfg_scale in cfg.evaluation.cfg_scales:
            for eval_integration_step in cfg.evaluation.eval_integration_steps:
                print(f"Eval: step={global_step}, cfg={cfg_scale}, steps={eval_integration_step}")
                metrics = evaluate(
                    rf=rf,
                    loader=loader,
                    device=cfg.device,
                    steps=eval_integration_step,
                    cfg_scale=cfg_scale,
                    n_batches=cfg.evaluation.eval_batches,
                    save_dir=eval_output_dir / "samples",
                    classifier=None,
                    epoch=i,
                    fid_model=fid_model,
                    fid_resizer=fid_resizer,
                    fid_stats=fid_stats,
                    real_feats=real_feats,
                    vae=vae,
                    save_gif=True,
                    save_samples=True
                )
                for key, value in metrics.items():
                    name = key.upper() if key == "fid" else key.replace("_", " ").title().replace(" ", "_")
                    full_name = f"Test/{name}/{cfg_scale}/{eval_integration_step}"
                    logger.add_scalar(full_name, value, global_step)
                    print(f"{full_name}: {value}")

        logger.flush()
    logger.close()


if __name__ == "__main__":
    main()

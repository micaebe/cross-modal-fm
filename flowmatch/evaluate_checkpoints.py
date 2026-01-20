import torch
import re
from config import build_rf
from dataset.build_dataset import build_dataloaders
from utils import set_seed, load_checkpoint
from evaluation.evaluation_utils import get_fid_components, get_real_features_for_dataset, get_vae, evaluate
from config import build_rf
from torch.utils.tensorboard import SummaryWriter
import hydra
from pathlib import Path
from omegaconf import DictConfig, open_dict
from hydra.core.hydra_config import HydraConfig

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.allow_tf32 = True

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    if "evaluation" not in cfg:
        print("Warning: 'evaluation' not found in config. Using defaults.")
        with open_dict(cfg):
            cfg.evaluation = {
                "eval_batches": 50,
                "eval_batch_size": 100,
                "use_test_set": True,
                "only_final": False,
                "cfg_scales": [1.0],
                "eval_integration_steps": [40]
            }

    set_seed(cfg.seed)
    
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    logger = SummaryWriter(log_dir=run_dir)

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

    if not "checkpoint_dir" in cfg:
        print("No checkpoint_dir provided")
        return

    checkpoint_paths = sorted(
        Path(cfg.checkpoint_dir).glob("*.pt"),
        key=lambda p: int(re.search(r'\d+', p.name).group())
    )
    if cfg.evaluation.only_final:
        checkpoint_paths = checkpoint_paths[-1:]

    print(f"Found {len(checkpoint_paths)} checkpoints to evaluate in {cfg.checkpoint_dir}")

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
                    save_dir=run_dir / "evaluation_samples",
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

                settings_tag = f"cfg{cfg_scale}_steps{eval_integration_step}"
                for key, value in metrics.items():
                    name = key.upper() if key == "fid" else key.replace("_", " ").title().replace(" ", "_")
                    tag = f"Test/{name}/{settings_tag}"
                    logger.add_scalar(tag, value, global_step)

    logger.flush()

if __name__ == "__main__":
    main()

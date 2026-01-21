import time
import torch
import hydra
from pathlib import Path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from torch.utils.tensorboard import SummaryWriter
from .config import build_rf, setup
from .utils import set_seed, save_checkpoint
from .train import train_rf
from .evaluation.evaluation_utils import get_fid_components, get_real_features_for_dataset, get_vae, evaluate


torch.set_float32_matmul_precision('high')
torch.backends.cudnn.allow_tf32 = True


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    logger = SummaryWriter(log_dir=run_dir)

    rf = build_rf(cfg)
    ema, optimizer, scheduler, global_step, train_loader, test_loader, classifier = setup(rf, cfg)
    print(f"Device: {cfg.device} | Dataset: {cfg.dataset.name} | Source: {cfg.mode.source} | Target: {cfg.mode.target} | Bidirectional: {cfg.mode.bidirectional} | Use Conditioning: {cfg.use_conditioning} | Label Embedding: {cfg.embeddings.name} | Embedding Std Scale: {cfg.embeddings.std_scale} | Use LN: {cfg.rf.ln} | LN Loc: {cfg.rf.ln_loc} | LN Scale: {cfg.rf.ln_scale} | Use mixed preicision: {cfg.mixed_precision}")
    total_params = sum(p.numel() for p in rf.model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params}")

    print("Setting up evaluation metrics...")
    vae = get_vae(cfg)
    fid_model, fid_resizer, fid_stats = get_fid_components(cfg.dataset.name, cfg.device)
    if vae:
        vae.to(cfg.device)
    real_feats = get_real_features_for_dataset(test_loader, fid_model, fid_resizer, cfg.device, max_batches=cfg.eval_batches, vae=vae)
    if vae:
        vae.to("cpu")

    print("Starting training...")
    total_outer_steps = cfg.total_steps // cfg.eval_every_steps
    start_outer_step = global_step // cfg.eval_every_steps
    data_iterator = iter(train_loader)

    for outer_step in range(start_outer_step, total_outer_steps):
        start_time = time.time()
        train_loss, global_step = train_rf(
            rf=rf,
            ema=ema,
            data_iterator=data_iterator,
            loader=train_loader,
            optimizer=optimizer,
            device=cfg.device,
            use_bf16=cfg.mixed_precision,
            num_steps=cfg.eval_every_steps,
            max_grad_norm=cfg.max_grad_norm,
            scheduler=scheduler,
            logger=logger,
            global_step=global_step
        )
        end_time = time.time()
        outer_step_time = end_time - start_time
        per_iter_time = outer_step_time / len(train_loader)
        if (outer_step + 1) % cfg.checkpoint_every_steps == 0 or outer_step == total_outer_steps - 1:
            save_checkpoint(rf.model, ema, optimizer, scheduler, global_step, run_dir)

        ema.apply_shadow(rf.model)
        metrics = evaluate(
            rf=rf,
            loader=test_loader,
            device=cfg.device,
            steps=cfg.eval_integration_steps,
            cfg_scale=cfg.eval_cfg_scale,
            n_batches=cfg.eval_batches,
            num_classes=cfg.dataset.num_classes,
            save_dir=run_dir / "eval_samples",
            classifier=classifier,
            epoch=outer_step,
            fid_model=fid_model,
            fid_resizer=fid_resizer,
            fid_stats=fid_stats,
            real_feats=real_feats,
            vae=vae,
        )
        ema.restore(rf.model)

        logger.add_scalar("Train/Avg_Loss", train_loss, global_step)
        logger.add_scalar("Test/Acc_L2", metrics["acc_l2"], global_step)
        logger.add_scalar("Test/Acc_Cos", metrics["acc_cos"], global_step)
        logger.add_scalar("Test/Acc_Class", metrics["acc_class"], global_step)
        logger.add_scalar("Test/Mean_L2", metrics["mean_l2"], global_step)
        logger.add_scalar("Test/Mean_Cos", metrics["mean_cos"], global_step)
        
        if "fid" in metrics:
             logger.add_scalar("Test/FID", metrics["fid"], global_step)
        if "precision" in metrics:
             logger.add_scalar("Test/Precision", metrics["precision"], global_step)
             logger.add_scalar("Test/Recall", metrics["recall"], global_step)
             logger.add_scalar("Test/Density", metrics["density"], global_step)
             logger.add_scalar("Test/Coverage", metrics["coverage"], global_step)

        print(f"Outer Step {outer_step:03d} | train_loss={train_loss:.4f} | "
              f"FID={metrics.get('fid', float('nan')):.2f} | "
              f"Precision={metrics.get('precision', float('nan')):.2f} | "
              f"Recall={metrics.get('recall', float('nan')):.2f} | "
              f"test_acc_l2={metrics['acc_l2']:.4f} | test_acc_cos={metrics['acc_cos']:.4f} | "
              f"test_acc_class={metrics['acc_class']:.4f} | "
              f"outer_step_time={outer_step_time:.2f}s | iter_time={per_iter_time:.4f}s")
    logger.flush()

if __name__ == "__main__":
    main()

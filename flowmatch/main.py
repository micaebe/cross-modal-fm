import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataset.build_dataset import build_dataloaders, get_dataset_info
from models.utils import EMA
from config import build_rf, setup_run
from utils import set_seed, load_checkpoint, save_checkpoint
from train import train_rf, evaluate, RF
from evaluation.train_classifier import Classifier
from evaluation.evaluation_utils import get_fid_components, get_real_features_for_dataset
from config import parse_args, build_rf
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import time

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.allow_tf32 = True


def main():
    args = parse_args()
    run_dir, args = setup_run(args)
    set_seed(args.seed)
    logger = SummaryWriter(log_dir=run_dir)

    # setup model, ema, optimizer, dataloaders
    rf = build_rf(args)
    ema = EMA(rf.model, decay=args.ema_decay, warmup_steps=args.ema_warmup_steps)
    optimizer = torch.optim.AdamW(rf.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, args.adam_beta2))
    train_loader, test_loader = build_dataloaders(args)
    print(f"Device: {args.device} | Dataset: {args.dataset} | Source: {args.source} | Target: {args.target} | Bidirectional: {args.bidirectional} | Use Conditioning: {args.use_conditioning} | Label Embedding: {args.label_embedding} | Embedding Std Scale: {args.embedding_std_scale} | Use LN: {args.use_ln} | LN Loc: {args.ln_loc} | LN Scale: {args.ln_scale} | Use mixed preicision: {args.mixed_precision}")

    scheduler = None
    if args.lr_warmup_steps > 0:
        warmup_steps = args.lr_warmup_steps
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0
        scheduler = LambdaLR(optimizer, lr_lambda)

    if args.resume_checkpoint:
        global_step = load_checkpoint(rf.model, ema, optimizer, scheduler, args)
        global_step += 1
    else:
        global_step = 0


    total_params = sum(p.numel() for p in rf.model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params}")

    print("Setting up evaluation metrics...")
    classifier = None
    if args.classifier_path:
        _, _, C, _ = get_dataset_info(args)
        classifier = Classifier(in_channels=C).to(args.device)
        classifier.load_state_dict(torch.load(args.classifier_path, map_location=args.device))
        classifier.eval()
    fid_model, fid_resizer, fid_stats = get_fid_components(args.dataset, args.device)
    real_feats = get_real_features_for_dataset(test_loader, fid_model, fid_resizer, args.device, max_batches=args.eval_batches)

    print("Starting training...")
    total_epochs = args.total_steps // args.checkpoint_every_steps
    start_epoch = global_step // args.checkpoint_every_steps
    data_iterator = iter(train_loader)
    for epoch in range(start_epoch, total_epochs):
        start_time = time.time()
        train_loss, global_step = train_rf(
            rf=rf,
            ema=ema,
            data_iterator=data_iterator,
            loader=train_loader,
            optimizer=optimizer,
            device=args.device,
            use_bf16=args.mixed_precision,
            num_steps=args.checkpoint_every_steps,
            grad_accum_steps=args.grad_accum_steps,
            scheduler=scheduler,
            logger=logger,
            global_step=global_step
        )
        end_time = time.time()
        epoch_time = end_time - start_time
        per_iter_time = epoch_time / len(train_loader)

        save_checkpoint(rf.model, ema, optimizer, scheduler, global_step, run_dir)

        metrics = evaluate(
            rf=rf,
            ema=ema,
            loader=test_loader,
            device=args.device,
            steps=args.eval_integration_steps,
            n_batches=args.eval_batches,
            save_dir=run_dir / "eval_samples",
            classifier=classifier,
            epoch=epoch,
            fid_model=fid_model,
            fid_resizer=fid_resizer,
            fid_stats=fid_stats,
            real_feats=real_feats
        )
        
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

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
              f"test_acc_l2={metrics['acc_l2']:.4f} | test_acc_cos={metrics['acc_cos']:.4f} | "
              f"test_acc_class={metrics['acc_class']:.4f} | "
              f"FID={metrics.get('fid', float('nan')):.2f} | "
              f"epoch_time={epoch_time:.2f}s | iter_time={per_iter_time:.4f}s")

    
if __name__ == "__main__":
    main()


import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dit import DiT_Llama
from utils import set_seed, EMA
from train import train_one_epoch, evaluate, RF
from train_classifier import Classifier
import time

#torch.set_float32_matmul_precision('high')
#torch.backends.cudnn.allow_tf32 = True

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--source", type=str, default="label", choices=["label", "noise", "image"])
    parser.add_argument("--target", type=str, default="image", choices=["label", "noise", "image"])
    parser.add_argument("--label_embedding", type=str, default="rectangle", choices=["grayscale", "clip", "ortho", "rectangle"])
    parser.add_argument("--embedding_std_scale", type=float, default=0.1)
    parser.add_argument("--embedding_norm_mode", type=str, default="none", choices=["none", "mean"])
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional flow matching")
    parser.add_argument("--use_conditioning", action="store_true", help="Use class conditioning inside DiT")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar"])
    parser.add_argument("--classifier_path", type=str, default="", help="Path to pretrained classifier for evaluation")
    parser.add_argument("--use_bf16", action="store_true", help="Use automatic mixed precision training with bfloat16")
    parser.add_argument("--use_ln", action="store_true", help="Use logit normal time sampling")
    parser.add_argument("--ln_loc", type=float, default=0.0, help="Logit normal location parameter")
    parser.add_argument("--ln_scale", type=float, default=1.0, help="Logit normal scale parameter")
    parser.add_argument("--use_sin_cos", action="store_true", help="Use sine-cosine interpolation for time sampling")

    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--project", type=str, default="flowmatching_ablations_mnist_cifar")
    parser.add_argument("--save_full_gradient", action="store_true", help="Save full gradient logs during training for debugging")

    args = parser.parse_args()
    set_seed(args.seed)


    print(f"Device: {args.device} | Dataset: {args.dataset} | Source: {args.source} | Target: {args.target} | Bidirectional: {args.bidirectional} | Use Conditioning: {args.use_conditioning} | Label Embedding: {args.label_embedding} | Embedding Std Scale: {args.embedding_std_scale} | Embedding Norm Mode: {args.embedding_norm_mode} | Use LN: {args.use_ln} | LN Loc: {args.ln_loc} | LN Scale: {args.ln_scale} | Use SinCos: {args.use_sin_cos} | Use BF16: {args.use_bf16}")


    if args.dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(2),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        val_transform = transform
        image_dims = (32, 32, 1)
        train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=val_transform)
    elif args.dataset == "cifar":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        image_dims = (32, 32, 3)
        train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform)


    #train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    device = args.device
    H, W, C = image_dims
    num_classes = 10
    if not args.use_conditioning or args.bidirectional:
        # if we dont use the class embeddings for conditioning, we use them as indicator function for cfg
        num_classes = 2


    model = DiT_Llama(
        in_channels=C,
        input_size=H,
        patch_size=2,
        dim=64 if C == 1 else 128,
        n_layers=3,
        n_heads=4,
        class_dropout_prob=0.1 if args.use_conditioning else 0.0, # in case we use CrossFlow style conditioning, we drop the class information not inside the architecture but outside
        num_classes=num_classes,
        bidirectional=args.bidirectional
    )
    model.to(device)
    if device == "cuda":
        model = torch.compile(model)

    rf = RF(
        model=model,
        ln=args.use_ln,
        ln_loc=args.ln_loc,
        ln_scale=args.ln_scale,
        use_sin_cos=args.use_sin_cos,
        source=args.source,
        target=args.target,
        embedding_type=args.label_embedding,
        emb_std_scale=args.embedding_std_scale,
        emb_norm_mode=args.embedding_norm_mode,
        img_dim=image_dims,
        bidirectional=args.bidirectional
    )
    rf.label_embedder.to(device)
    ema = EMA(model, decay=0.999, warmup_steps=1000)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    run_name = f"{args.dataset}_{args.source}_to_{args.target}_{args.use_conditioning}_{args.bidirectional}"
    if args.source == "label" or args.target == "label":
        run_name += f"_{args.label_embedding}_std{str(args.embedding_std_scale).replace('.', '')}_{args.embedding_norm_mode}"
    if args.use_ln:
        run_name += f"_ln_loc{str(args.ln_loc).replace('.', '')}_scale{str(args.ln_scale).replace('.', '')}"
    if args.use_sin_cos:
        run_name += f"_sincos"
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params}")
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(project=args.project, name=run_name, config=vars(args))
        wandb.summary["params"] = total_params

    classifier = None
    if args.classifier_path:
        classifier = Classifier(in_channels=C).to(device)
        classifier.load_state_dict(torch.load(args.classifier_path, map_location=device))
        classifier.eval()

    print("Starting training...")

    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = train_one_epoch(
            rf=rf,
            ema=ema,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            use_conditioning=args.use_conditioning,
            log_wandb=(args.wandb and WANDB_AVAILABLE),
            log_full_gradient=args.save_full_gradient,
            use_bf16=args.use_bf16,
            epoch=epoch,
        )
        end_time = time.time()
        epoch_time = end_time - start_time
        per_iter_time = epoch_time / len(train_loader)
        if epoch % 20 == 0:
            test_acc_l2, test_acc_cos, test_acc_class = evaluate(
                rf=rf,
                ema=ema,
                loader=test_loader,
                device=device,
                steps=40,
                use_conditioning=args.use_conditioning,
                save_dir=f"./training_outputs/{run_name}",
                classifier=classifier,
                epoch=epoch
            )
            if args.wandb and WANDB_AVAILABLE:
                wandb.log({"epoch": epoch, "train_loss": train_loss, "test_acc_l2": test_acc_l2, "test_acc_cos": test_acc_cos, "test_acc_class": test_acc_class})
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | test_acc_l2={test_acc_l2:.4f} | test_acc_cos={test_acc_cos:.4f} | test_acc_class={test_acc_class:.4f} | epoch_time={epoch_time:.2f}s | iter_time={per_iter_time:.4f}s")
        else:
            if args.wandb and WANDB_AVAILABLE:
                wandb.log({"epoch": epoch, "train_loss": train_loss})
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | epoch_time={epoch_time:.2f}s | iter_time={per_iter_time:.4f}s")

    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()
    

if __name__ == "__main__":
    main()
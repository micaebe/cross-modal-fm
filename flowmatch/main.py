
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dit import DiT_Llama
from utils import set_seed, EMA
from train import train_one_epoch, evaluate, RF

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--source", type=str, default="label", choices=["label", "noise", "image"])
    parser.add_argument("--target", type=str, default="image", choices=["label", "noise", "image"])
    parser.add_argument("--label_embedding", type=str, default="ortho", choices=["grayscale", "clip", "ortho", "rectangle"])
    parser.add_argument("--embedding_std_scale", type=float, default=0.1)
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional flow matching")
    parser.add_argument("--use_conditioning", action="store_true", help="Use class conditioning inside DiT")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar"])



    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--project", type=str, default="mnist_flow_matching3")

    args = parser.parse_args()
    set_seed(args.seed)


    if args.dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Pad(2),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        image_dims = (32, 32, 1)
        train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    elif args.dataset == "cifar":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        image_dims = (32, 32, 3)
        train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)


    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

    device = args.device
    H, W, C = image_dims
    num_classes = 10
    if not args.use_conditioning:
        num_classes = 0
    if args.bidirectional:
        num_classes = 2


    model = DiT_Llama(
        in_channels=C,
        input_size=H,
        patch_size=2,
        dim=64 if C == 1 else 128,
        n_layers=6,
        n_heads=4,
        class_dropout_prob=0.0,
        num_classes=num_classes
    )
    rf = RF(
        model=model,
        ln=False,
        source=args.source,
        target=args.target,
        embedding_type=args.label_embedding,
        emb_std_scale=args.embedding_std_scale,
        img_dim=image_dims,
        bidirectional=args.bidirectional
    )
    model.to(device)
    rf.label_embedder.to(device)
    ema = EMA(model, decay=0.995)
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params}")

    if args.wandb and WANDB_AVAILABLE:
        name = f"{args.source}_to_{args.target}_{args.label_embedding}_{args.dataset}"
        wandb.init(project=args.project, name=name, config=vars(args))
        wandb.summary["params"] = total_params

    print("Starting training...")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            rf=rf,
            ema=ema,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            use_conditioning=args.use_conditioning,
            log_wandb=(args.wandb and WANDB_AVAILABLE),
            log_full_gradient=False,
            epoch=epoch
        )
        test_acc_l2, test_acc_cos = evaluate(
            rf=rf,
            ema=ema,
            loader=test_loader,
            device=device,
            steps=50,
            use_conditioning=args.use_conditioning,
            epoch=epoch
        )
        
        if args.wandb and WANDB_AVAILABLE:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "test_acc_l2": test_acc_l2, "test_acc_cos": test_acc_cos})
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | test_acc_l2={test_acc_l2:.4f} | test_acc_cos={test_acc_cos:.4f}")

    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()

if __name__ == "__main__":
    main()

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from flowmatch.models import MLP, UnetWrapper
from torchcfm.models.unet import UNetModel
from flowmatch.label_embeddings import make_embedding_provider
from flowmatch.data import get_datasets
from flowmatch.utils import set_seed, EMA
from flowmatch.train import train_one_epoch, evaluate

try:
    import wandb
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=95)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--v_pred", action="store_true", help="Use v-prediction")

    parser.add_argument("--model", type=str, default="unet", choices=["unet", "mlp"])
    parser.add_argument("--embedding_provider", type=str, default="scalar", choices=["scalar", "clip", "circle", "random"])
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist"])

    parser.add_argument("--direction", type=str, default="label_to_image",
                        choices=["image_to_label", "label_to_image"])
    parser.add_argument("--sigma_min", type=float, default=0.0)

    parser.add_argument("--int_steps", type=int, default=40)

    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--project", type=str, default="mnist_flow_matching3")
    parser.add_argument("--run_name", type=str, default="label2image")

    args = parser.parse_args()
    set_seed(args.seed)


    train_ds, test_ds, (C, H, W), num_classes = get_datasets(args.dataset)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)

    device = args.device
    model = None
    if args.model == "unet":
        #model = UNet(in_ch=C, base_ch=64, t_dim=8).to(device)
        #u_model = UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1)
        u_model = UNetModel(
                dim=(1, 28, 28),
                num_channels=32,
                num_res_blocks=1,
                channel_mult=(1,),
                attention_resolutions="7",
                use_scale_shift_norm=True
            )
        model = UnetWrapper(u_model).to(device)
    elif args.model == "mlp":
        model = MLP(in_shape=(C, H, W), out_shape=(C, H, W), hidden_dim=1024, t_dim=8).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    ema = EMA(model, decay=0.99)
    

    embedding_provider = make_embedding_provider(
        args.embedding_provider,
        H=H, W=W, C=C,
        num_classes=num_classes,
        device=device
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params}")


    if args.wandb and WANDB_AVAILABLE:
        wandb.init(project=args.project, name=args.run_name, config=vars(args))
        wandb.summary["params"] = total_params

    print("Starting training...")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model=model,
            ema=ema,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            embedding_provider=embedding_provider, 
            direction=args.direction,
            criterion=criterion,
            log_wandb=(args.wandb and WANDB_AVAILABLE),
            epoch=epoch,
            v_pred=True
        )

        test_acc_l2, test_acc_cos, fig1, fig2, fig3 = evaluate(
            model=model,
            ema=ema,
            loader=test_loader,
            device=device,
            embedding_provider=embedding_provider,
            direction=args.direction,
            steps=args.int_steps,
            v_pred=True
        )

        if args.wandb and WANDB_AVAILABLE:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "test_acc_l2": test_acc_l2, "test_acc_cos": test_acc_cos})
            wandb.log({"label_sequence": wandb.Image(fig1, caption="Image to Label Sequence")})
            wandb.log({"image_sequence": wandb.Image(fig2, caption="Label to Image Sequence")})
            wandb.log({"image_sequence_2": wandb.Image(fig3, caption="Label to Image Sequence 2")})
        plt.close(fig1)
        plt.close(fig2)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | test_acc_l2={test_acc_l2:.4f} | test_acc_cos={test_acc_cos:.4f}")

    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()

if __name__ == "__main__":
    main()
import argparse
from pathlib import Path
import json
import torch
from datetime import datetime
from models.dit import DiT_Llama
from models.fast_dit import DiT
from models.unet import UNetModel
from dataset.build_dataset import get_dataset_info
from embeddings.build_embeddings import build_embedding_provider
from utils import load_checkpoint
from rf import RF

# probably would make sense to use e.g. hydra
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--project_name", type=str, default="runs")

    # Model args
    parser.add_argument("--model", type=str, default="DiT", choices=["DiT_Llama", "DiT", "UNet"], help="Model architecture")
    parser.add_argument("--dim", type=int, default=128, help="Model dimension (in case of UNet its the number of channels)")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=2, help="Number of attention heads (only for transformer based models)")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size (only for transformer based models)")
    parser.add_argument("--channel_mult", type=str, default="1,2,4,8", help="Channel multipliers for UNet, eg. '1,2,4,8'")
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional flow matching")
    parser.add_argument("--lambda_b", type=float, default=0.5, help="Weighting for bidirectional flow matching (between 0 and 1, where the endpoints are unidirectional models)")
    parser.add_argument("--cls_dropout", type=float, default=0.1, help="Dropout rate for class conditioning")

    # Training args
    parser.add_argument("--total_steps", type=int, default=100_000, help="Total number of training steps.")
    parser.add_argument("--checkpoint_every_steps", type=int, default=2000, help="Checkpointing and evaluation frequency in steps.")
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=1000, help="Number of lr warmup steps")

    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta2", type=float, default=0.995)
    parser.add_argument("--ema_decay", type=float, default=0.9995)
    parser.add_argument("--ema_warmup_steps", type=int, default=1000, help="Number of warmup steps for EMA")

    parser.add_argument("--compile_model", action="store_true", help="Use torch.compile on the model")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training with bfloat16")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps (1 = no accumulation)")

    # Eval args (eval during training)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--eval_batches", type=int, default=30)
    parser.add_argument("--eval_cfg_scale", type=float, default=1.0)
    parser.add_argument("--eval_integration_steps", type=int, default=40)

    # Data args
    parser.add_argument("--source", type=str, default="label", choices=["label", "noise", "image"])
    parser.add_argument("--target", type=str, default="image", choices=["label", "noise", "image"])
    parser.add_argument("--label_embedding", type=str, default="rectangle", choices=["grayscale", "clip", "rectangle", "low_rank"])
    parser.add_argument("--embedding_std_scale", type=float, default=0.1)
    parser.add_argument("--use_conditioning", action="store_true", help="Use class conditioning inside DiT")

    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar", "imagenet100"])
    parser.add_argument("--data_dir", default="./flowmatch/vae_mds_100", help="Path to imagenet dataset")

    # Path args
    parser.add_argument("--use_ln", action="store_true", help="Use logit normal time sampling")
    parser.add_argument("--ln_loc", type=float, default=0.0, help="Logit normal location parameter")
    parser.add_argument("--ln_scale", type=float, default=1.0, help="Logit normal scale parameter")


    parser.add_argument("--classifier_path", type=str, default="", help="Path to pretrained classifier for evaluation (optional)")
    parser.add_argument("--fid_ref_dir", type=str, default=None, help="Path to reference images for FID calculation (imagenet)")
    parser.add_argument("--project", type=str, default="")
    parser.add_argument("--resume_checkpoint", type=str, default="", help="Path to checkpoint to resume from")


    return parser.parse_args()


def setup_run(args):
    if args.resume_checkpoint:
        run_dir = Path(args.resume_checkpoint).parent
        print(f"Resuming experiment in: {run_dir}")
        with open(run_dir / "config.json", "r") as f:
            checkpoint_args = json.load(f)
        preserve_keys = ["resume_checkpoint", "device", "total_steps"]
        for k, v in checkpoint_args.items():
            if k not in preserve_keys:
                setattr(args, k, v)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        noise_str = str(args.embedding_std_scale).replace(".", "_")
        run_name = f"{timestamp}_{args.dataset}_{noise_str}"
        if not args.bidirectional:
            run_name += f"_{args.source}_{args.target}"
        else:
            run_name += "_bidir"
        run_dir = Path(args.project_name) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Starting new experiment in: {run_dir}")
        with open(run_dir / "config.json", "w") as f:
            json.dump(vars(args), f, indent=4)
    return run_dir, args



def _get_model_internal_cls_dropout_prob(args):
    # if we have as source or target "label"
    # then we apply the class dropout outside of the model
    if args.source == "label" or args.target == "label":
        return 0.0
    return args.class_dropout_prob

def build_rf(args):
    H, W, C, num_classes = get_dataset_info(args)
    num_classes_model = num_classes
    if not args.use_conditioning:
        # if we dont use the class embeddings for conditioning, we use them as indicator function for cfg
        num_classes_model = 2
    cls_dropout_prob_model = _get_model_internal_cls_dropout_prob(args)

    if args.model == "DiT_Llama":
        model = DiT_Llama(
            in_channels=C,
            input_size=H,
            patch_size=args.patch_size,
            dim=args.dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            class_dropout_prob=cls_dropout_prob_model,
            num_classes=num_classes_model,
            bidirectional=args.bidirectional,
        )
    elif args.model == "DiT":
        model = DiT(
            in_channels=C,
            input_size=H,
            patch_size=args.patch_size,
            hidden_size=args.dim,
            depth=args.n_layers,
            num_heads=args.n_heads,
            class_dropout_prob=cls_dropout_prob_model,
            num_classes=num_classes_model,
            bidirectional=args.bidirectional,
        )
    elif args.model == "UNet":
       model = UNetModel(
           image_size=H,
           in_channels=C,
           model_channels=args.dim,
           out_channels=C,
           num_res_blocks=args.n_layers,
           num_classes=num_classes_model,
           channel_mult=tuple(map(int, args.channel_mult.split(","))),
           attention_resolutions=[2, 4],
           bidirectional=args.bidirectional,
       )
    device = args.device
    model.to(device)

    if device == "cuda" and args.compile_model:
        model = torch.compile(model)

    label_embedder = None
    if args.source != "noise" and args.target != "noise":
        # if dataset is imagenet100, we have 4 codes, if its mnist or cifar we have 1 code
        codes_per_cell = 4
        if args.dataset in ["mnist", "cifar"]:
            codes_per_cell = 2

        label_embedder = build_embedding_provider(args.label_embedding,
                                                  H=H,
                                                  W=W,
                                                  C=C,
                                                  num_classes=num_classes,
                                                  std_scale=args.embedding_std_scale,
                                                  blur_sigma=1.0,                   # only used if embedding is "rectangle"
                                                  codes_per_cell=codes_per_cell)    # only used if embedding is "rectangle"
        label_embedder.to(device)

    rf = RF(
        model=model,
        ln=args.use_ln,
        ln_loc=args.ln_loc,
        ln_scale=args.ln_scale,
        source=args.source,
        target=args.target,
        label_embedder=label_embedder,
        img_dim=(H, W, C),
        lambda_b=args.lambda_b,
        bidirectional=args.bidirectional,
        cfg_dropout_prob=args.cls_dropout,
        use_conditioning=args.use_conditioning
    )
    return rf

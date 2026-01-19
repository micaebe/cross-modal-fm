import os
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import torch
from torch.utils.data import DataLoader

from torch_fidelity import calculate_metrics
from torch_fidelity.generative_model_base import GenerativeModelBase

from flowmatch.rf import RF
from flowmatch.dit import DiT_Llama
from flowmatch.train import (
    get_final_label_and_image,
    count_correct_image_label_generations,
)


# -------------------------
# Config inference (from dirname)
# -------------------------
@dataclass
class RunConfig:
    dataset: str = "cifar10"          # cifar10 | mnist | imagenet (optional)
    source: str = "label"             # label | image | noise
    target: str = "image"             # label | image | noise
    use_conditioning: bool = True
    bidirectional: bool = True

    embedding_type: str = "rectangle" # grayscale | rectangle | ortho | clip
    emb_std_scale: float = 0.2
    emb_norm_mode: str = "none"

    use_ln: bool = False
    ln_loc: float = 0.0
    ln_scale: float = 1.0
    use_sin_cos: bool = False


def parse_config_from_dirname(dirname: str) -> RunConfig:
    """
    Expected pattern (your old script style):
      <dataset>_<source>_to_<target>_<use_conditioning>_<bidirectional>_<embedding_type>_stdX_<norm>
    Example:
      cifar10_label_to_image_True_True_rectangle_std2_none
    """
    parts = dirname.split("_")
    cfg = RunConfig()

    # minimal guard
    if len(parts) < 6:
        return cfg

    # dataset is first token
    cfg.dataset = parts[0]

    # source / to / target
    # idx: 1=source, 2=to, 3=target
    cfg.source = parts[1]
    cfg.target = parts[3]

    # conditioning / bidirectional
    # idx: 4=use_conditioning, 5=bidirectional
    cfg.use_conditioning = (parts[4] == "True")
    cfg.bidirectional = (parts[5] == "True")

    # optional embedding params (only if not noise)
    if cfg.source != "noise" and len(parts) >= 9:
        cfg.embedding_type = parts[6]

        std_str = parts[7]  # e.g. std2, std0.2
        if std_str.startswith("std"):
            val_str = std_str.replace("std", "")
            try:
                if "." in val_str:
                    cfg.emb_std_scale = float(val_str)
                else:
                    # historical: std2 => 0.2
                    cfg.emb_std_scale = float(val_str) / 10.0
            except ValueError:
                pass

        cfg.emb_norm_mode = parts[8]

    return cfg


def infer_run_config_from_ckpt(ckpt_path: str) -> RunConfig:
    parent = os.path.basename(os.path.dirname(os.path.abspath(ckpt_path)))
    return parse_config_from_dirname(parent)


# -------------------------
# Dataset / Model helpers
# -------------------------
def get_img_spec(dataset: str) -> Tuple[int, int, int]:
    ds = dataset.lower()
    if ds == "mnist":
        return 28, 28, 1
    # default CIFAR10-like
    return 32, 32, 3


def default_fid_ref(dataset: str) -> Optional[str]:
    """
    torch-fidelity built-in dataset names vary; CIFAR10 is reliable.
    For others, you can pass a path or name via --fid_ref.
    """
    ds = dataset.lower()
    if ds == "cifar10":
        return "cifar10-val"
    # For MNIST/ImageNet, require explicit --fid_ref (or a path).
    return None


def build_dataloader(dataset: str, batch_size: int) -> DataLoader:
    ds = dataset.lower()
    if ds == "mnist":
        from torchvision.datasets import MNIST
        from torchvision import transforms
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        dset = MNIST(root="./data", train=False, download=True, transform=tfm)
    elif ds == "cifar10":
        from torchvision.datasets import CIFAR10
        from torchvision import transforms
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])
        dset = CIFAR10(root="./data", train=False, download=True, transform=tfm)
    else:
        raise ValueError(
            f"Unsupported dataset='{dataset}'. Use --dataset cifar10|mnist or "
            f"extend build_dataloader()."
        )

    return DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


def build_model(cfg: RunConfig, device: str) -> DiT_Llama:
    H, W, C = get_img_spec(cfg.dataset)

    # num_classes logic follows your old scripts:
    # - DiT has an extra "null" class for conditioning => 11 for CIFAR10
    # - CrossFlow indicator mode => 2 classes (0 conditioned, 1 unconditioned)
    # - bidirectional often also uses indicator mode => 2
    if cfg.use_conditioning:
        num_classes = 11 if cfg.dataset.lower() in ("cifar10", "mnist") else 1001
    else:
        num_classes = 2
    if cfg.bidirectional:
        num_classes = 2

    dim = 64 if C == 1 else 256
    n_layers = 6 if C == 1 else 10
    patch_size = 2

    model = DiT_Llama(
        in_channels=C,
        input_size=H,
        patch_size=patch_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=8,
        class_dropout_prob=0.0,
        num_classes=num_classes,
        bidirectional=cfg.bidirectional,
    ).to(device)
    return model


def load_compiled_checkpoint(model: torch.nn.Module, ckpt_path: str) -> torch.nn.Module:
    sd = torch.load(ckpt_path, map_location="cpu")
    new_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=True)
    return model


# -------------------------
# FID/PRDC Wrapper
# -------------------------
class FMFIDWrapper(GenerativeModelBase):
    """
    torch-fidelity expects a GenerativeModelBase that outputs uint8 images in [0,255], shape [B,C,H,W].
    We generate images from the RF in the specified direction for FID/PRDC.

    This wrapper supports both conditioning styles:
      - use_conditioning=True: cond=y, null_cond=num_classes
      - use_conditioning=False (CrossFlow indicators): cond=0, null_cond=1
    """
    def __init__(
        self,
        rf: RF,
        num_classes: int,
        image_size: int,
        channels: int,
        device: str,
        steps: int,
        cfg: float,
        direction: str,
        use_conditioning: bool,
    ):
        super().__init__()
        self.rf = rf
        self._num_classes = num_classes
        self.image_size = image_size
        self.channels = channels
        self.device = device
        self.steps = steps
        self.cfg = cfg
        self.direction = direction
        self.use_conditioning = use_conditioning

    @property
    def z_size(self): return 1

    @property
    def z_type(self): return "normal"

    @property
    def num_classes(self): return self._num_classes

    @torch.no_grad()
    def forward(self, x, y=None):
        # y is class label on CPU for torch-fidelity
        if y is None:
            y = torch.randint(0, self.num_classes, (x.size(0),), device="cpu").long()

        if self.rf.source_type != "noise":
            # label embeddings (sample in image-space tensor)
            z = self.rf.label_embedder.sample(y, sample=True).to(self.device)
        else:
            z = torch.randn(x.size(0), self.channels, self.image_size, self.image_size, device=self.device)

        if self.use_conditioning:
            cond = y.to(self.device)
            null_cond = torch.full_like(y, self.rf.label_embedder.num_classes).to(self.device)
        else:
            # CrossFlow-style indicators
            cond = torch.zeros_like(y).to(self.device)
            null_cond = torch.ones_like(y).to(self.device)

        traj = self.rf.sample(
            z, cond, null_cond,
            sample_steps=self.steps,
            cfg=self.cfg,
            direction=self.direction,
        )

        imgs = traj[-1].float()
        # map [-1,1] -> [0,1] if needed
        if imgs.min() < 0:
            imgs = (imgs + 1.0) / 2.0

        return (imgs.clamp(0, 1) * 255.0).to(torch.uint8)


# -------------------------
# Core evaluation per direction
# -------------------------
@torch.no_grad()
def eval_label_metrics(
    rf: RF,
    loader: DataLoader,
    device: str,
    steps: int,
    cfg: float,
    use_conditioning: bool,
    classifier=None,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Computes:
      - image classifier accuracy (if classifier provided and final_image exists)
      - label accuracy via nearest-labels (L2 & cosine) if final_label exists
    Uses cfg and steps for sampling, so metrics correspond to the given (cfg, nfe).
    """
    total_class = total_l2 = total_cos = total = 0

    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if use_conditioning:
            cond = y
            null_cond = torch.full_like(y, rf.label_embedder.num_classes)
        else:
            cond = torch.zeros_like(y).long()
            null_cond = torch.ones_like(y).long()

        x0 = rf.get_distribution(x, y, "source")
        x1 = rf.get_distribution(x, y, "target")

        fwd = rf.sample(x0, cond, null_cond, sample_steps=steps, cfg=cfg, direction="forward")
        bwd = rf.sample(x1, cond, null_cond, sample_steps=steps, cfg=cfg, direction="backward")

        label_final, image_final = get_final_label_and_image(rf, fwd, bwd)
        c_cls, c_l2, c_cos = count_correct_image_label_generations(
            rf, image_final, label_final, classifier, y
        )

        total_class += c_cls
        total_l2 += c_l2
        total_cos += c_cos
        total += y.numel()

        if max_batches is not None and i + 1 >= max_batches:
            break

    return {
        "acc_class": total_class / max(1, total),
        "acc_l2": total_l2 / max(1, total),
        "acc_cos": total_cos / max(1, total),
    }


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        "Checkpoint evaluation: CFG/NFE sweep, both directions, FID/PRDC + label metrics."
    )

    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (e.g., model_emaXXX.pt).")
    parser.add_argument("--cfg_scales", type=float, nargs="+", required=True, help="One or more CFG scales.")
    parser.add_argument("--nfes", type=int, nargs="+", required=True, help="One or more NFE/ODE steps.")
    parser.add_argument("--out", type=str, default="eval_results.jsonl", help="Output jsonl path.")

    # dataset & fid ref
    parser.add_argument("--dataset", type=str, default="auto", help="auto|cifar10|mnist")
    parser.add_argument("--fid_ref", type=str, default="auto",
                        help="auto|torch-fidelity dataset name|path to images. Default: cifar10-val for cifar10.")
    parser.add_argument("--num_samples", type=int, default=10000, help="Num samples for torch-fidelity input1.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for loader and torch-fidelity.")
    parser.add_argument("--max_batches", type=int, default=None, help="Optional: limit loader batches for quick test.")

    # conditioning style
    parser.add_argument("--use_conditioning", action="store_true", help="Use class conditioning (DiT class tokens).")
    parser.add_argument("--use_crossflow_indicators", action="store_true",
                        help="Use CrossFlow-style indicator conditioning (cond=0, null=1).")

    # overrides (optional, but helps when dirname parsing doesn't match)
    parser.add_argument("--embedding_type", type=str, default=None, help="Override: grayscale|rectangle|ortho|clip")
    parser.add_argument("--emb_std_scale", type=float, default=None, help="Override embedding std scale.")
    parser.add_argument("--emb_norm_mode", type=str, default=None, help="Override embedding norm mode.")
    parser.add_argument("--bidirectional", action="store_true", help="Override: model was trained bidirectional.")
    parser.add_argument("--source", type=str, default=None, help="Override source type.")
    parser.add_argument("--target", type=str, default=None, help="Override target type.")

    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # infer cfg from ckpt dirname
    inferred = infer_run_config_from_ckpt(args.ckpt)

    # apply overrides
    cfg = inferred
    if args.dataset != "auto":
        cfg.dataset = args.dataset
    if args.source is not None:
        cfg.source = args.source
    if args.target is not None:
        cfg.target = args.target

    # Conditioning selection:
    # - If user explicitly chooses CrossFlow indicators, use_conditioning=False
    # - Else if user explicitly chooses use_conditioning, use_conditioning=True
    # - Else fall back to inferred config
    if args.use_crossflow_indicators:
        cfg.use_conditioning = False
    elif args.use_conditioning:
        cfg.use_conditioning = True

    if args.bidirectional:
        cfg.bidirectional = True

    if args.embedding_type is not None:
        cfg.embedding_type = args.embedding_type
    if args.emb_std_scale is not None:
        cfg.emb_std_scale = args.emb_std_scale
    if args.emb_norm_mode is not None:
        cfg.emb_norm_mode = args.emb_norm_mode

    # fid reference
    if args.fid_ref == "auto":
        fid_ref = default_fid_ref(cfg.dataset)
        if fid_ref is None:
            raise ValueError(
                f"--fid_ref must be provided for dataset='{cfg.dataset}'. "
                f"For cifar10, auto uses 'cifar10-val'."
            )
    else:
        fid_ref = args.fid_ref

    # Build model & load weights
    model = build_model(cfg, device=device)
    model = load_compiled_checkpoint(model, args.ckpt)
    model.eval()

    # Build dataloader
    loader = build_dataloader(cfg.dataset, batch_size=args.batch_size)

    H, W, C = get_img_spec(cfg.dataset)

    # Create TWO RF instances to strictly evaluate BOTH directions:
    # 1) label -> image
    # 2) image -> label
    # This matches the requirement: "in both directions (image->label and label->image)".
    rf_l2i = RF(
        model=model,
        source="label",
        target="image",
        ln=cfg.use_ln,
        ln_loc=cfg.ln_loc,
        ln_scale=cfg.ln_scale,
        embedding_type=cfg.embedding_type,
        emb_std_scale=cfg.emb_std_scale,
        emb_norm_mode=cfg.emb_norm_mode,
        bidirectional=cfg.bidirectional,
        use_sin_cos=cfg.use_sin_cos,
        img_dim=(H, W, C),
    )
    rf_i2l = RF(
        model=model,
        source="image",
        target="label",
        ln=cfg.use_ln,
        ln_loc=cfg.ln_loc,
        ln_scale=cfg.ln_scale,
        embedding_type=cfg.embedding_type,
        emb_std_scale=cfg.emb_std_scale,
        emb_norm_mode=cfg.emb_norm_mode,
        bidirectional=cfg.bidirectional,
        use_sin_cos=cfg.use_sin_cos,
        img_dim=(H, W, C),
    )

    # torch-fidelity wrapper (only meaningful when generating images)
    fid_wrapper = FMFIDWrapper(
        rf=rf_l2i,
        num_classes=10,
        image_size=H,
        channels=C,
        device=device,
        steps=1,      # placeholder (set per nfe in loop)
        cfg=1.0,      # placeholder (set per cfg in loop)
        direction="forward",
        use_conditioning=cfg.use_conditioning,
    )

    results: List[Dict[str, Any]] = []

    for cfg_scale in args.cfg_scales:
        for nfe in args.nfes:
            # ---------------------------
            # Direction 1: label -> image
            # ---------------------------
            l2i_metrics = eval_label_metrics(
                rf=rf_l2i,
                loader=loader,
                device=device,
                steps=nfe,
                cfg=cfg_scale,
                use_conditioning=cfg.use_conditioning,
                classifier=None,
                max_batches=args.max_batches,
            )

            # FID/PRDC for label->image at this cfg/nfe
            fid_wrapper.steps = nfe
            fid_wrapper.cfg = cfg_scale

            fid_prdc = calculate_metrics(
                input1=fid_wrapper,
                input2=fid_ref,
                fid=True,
                prdc=True,
                cuda=(device == "cuda"),
                batch_size=args.batch_size,
                input1_model_num_samples=args.num_samples,
                cache=False,
            )

            rec_l2i = {
                "direction": "label_to_image",
                "cfg": float(cfg_scale),
                "nfe": int(nfe),
                "dataset": cfg.dataset,
                "ckpt": os.path.abspath(args.ckpt),
                **l2i_metrics,
                **fid_prdc,
            }
            results.append(rec_l2i)

            print(f"[L->I] cfg={cfg_scale} nfe={nfe} FID={fid_prdc.get('frechet_inception_distance', float('nan')):.3f} "
                  f"acc_l2={l2i_metrics['acc_l2']:.3f} acc_cos={l2i_metrics['acc_cos']:.3f}")

            # ---------------------------
            # Direction 2: image -> label
            # ---------------------------
            # No FID/PRDC here (not meaningful), but we DO compute label-accuracy/distances (cfg-aware).
            i2l_metrics = eval_label_metrics(
                rf=rf_i2l,
                loader=loader,
                device=device,
                steps=nfe,
                cfg=cfg_scale,
                use_conditioning=cfg.use_conditioning,
                classifier=None,
                max_batches=args.max_batches,
            )

            rec_i2l = {
                "direction": "image_to_label",
                "cfg": float(cfg_scale),
                "nfe": int(nfe),
                "dataset": cfg.dataset,
                "ckpt": os.path.abspath(args.ckpt),
                **i2l_metrics,
            }
            results.append(rec_i2l)

            print(f"[I->L] cfg={cfg_scale} nfe={nfe} acc_l2={i2l_metrics['acc_l2']:.3f} acc_cos={i2l_metrics['acc_cos']:.3f}")

    # write jsonl
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()

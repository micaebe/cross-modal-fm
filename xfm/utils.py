import torch
import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, ema, optimizer, scheduler, global_step, run_dir):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "global_step": global_step
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    else:
        checkpoint["scheduler_state_dict"] = None
    torch.save(checkpoint, run_dir / f"ckpt_{global_step}.pt")


def load_checkpoint(model, optimizer, ema, scheduler, checkpoint_path, load_ema_weights=False):
    """
    Load a checkpoint
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    global_step = checkpoint["global_step"]
    if model:
        if load_ema_weights:
            state_dict = checkpoint["ema_state_dict"]["shadow"]
            strict = False
        else:
            state_dict = checkpoint["model_state_dict"]
            strict = True
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        if hasattr(model, "_orig_mod"):
            model._orig_mod.load_state_dict(state_dict, strict=strict)
        else:
            model.load_state_dict(state_dict, strict=strict)
    if ema:
        ema.load_state_dict(checkpoint["ema_state_dict"])
        ema.step = global_step
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return global_step



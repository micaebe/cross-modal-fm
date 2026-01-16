import torch
import torch.nn.functional as F
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

# Label accuracy via nearest neighbor
def nearest_labels_l2(final_states, means):
    flat_states = final_states.flatten(start_dim=1)
    flat_means = means.flatten(start_dim=1)
    dists = torch.cdist(flat_states, flat_means, p=2)
    preds = torch.argmin(dists, dim=1)
    return preds, dists

def nearest_labels_cos(final_states, means):
    X = F.normalize(final_states.flatten(start_dim=1), p=2, dim=1, eps=1e-12)
    M = F.normalize(means.flatten(start_dim=1), p=2, dim=1, eps=1e-12)
    sims = X @ M.T
    preds = sims.argmax(dim=1)
    return preds, sims

def nearest_labels(final_states, means):
    preds_l2, dists_l2 = nearest_labels_l2(final_states, means)
    preds_cos, sims_cos = nearest_labels_cos(final_states, means)
    return preds_l2, preds_cos, dists_l2, sims_cos


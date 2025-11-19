import os
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_source_target(direction: str, image: Tensor, label: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    if direction == "image_to_label":
        return image, label, t
    elif direction == "label_to_image":
        # we also rotate t so that inputs are same
        return label, image, 1.0 - t
    else:
        raise ValueError(f"Unknown direction: {direction}")
    

def show_sequence(seq, ncols=10, nrows=10, vmin=-1, vmax=1, cmap="gray", title=None):
    if isinstance(seq, list):
        seq = torch.stack(seq, dim=0)
    T, B, C, H, W = seq.shape
    nrows = min(nrows, B)
    cols = min(ncols, T)
    step = max(1, T // ncols)

    fig, axes = plt.subplots(nrows, cols, figsize=(cols, nrows))
    for r in range(nrows):
        for c in range(cols):
            t_idx = c * step
            axes[r, c].imshow(seq[t_idx, r].permute(1,2,0).cpu().numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
            axes[r, c].axis("off")

    if title:
        fig.suptitle(title)
    return fig


def nearest_labels_l2(final_states: Tensor, means: Tensor) -> Tensor:
    diffs = final_states.unsqueeze(1) - means.unsqueeze(0)
    dists = (diffs**2).sum(dim=(2,3,4)).sqrt()
    preds = torch.argmin(dists, dim=1)
    return preds

def nearest_labels_cos(final_states: Tensor, means: Tensor) -> Tensor:
    X = F.normalize(final_states.flatten(start_dim=1), p=2, dim=1, eps=1e-12)
    M = F.normalize(means.flatten(start_dim=1), p=2, dim=1, eps=1e-12)
    sims = X @ M.T
    preds = sims.argmax(dim=1)  
    return preds

def nearest_labels(final_states: Tensor, means: Tensor) -> tuple[Tensor, Tensor]:
    preds_l2 = nearest_labels_l2(final_states, means)
    preds_cos = nearest_labels_cos(final_states, means)
    return preds_l2, preds_cos

def log_full_grads(model, epoch, batch_idx, save_dir="grads"):
    os.makedirs(save_dir, exist_ok=True)
    grad_dict = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_dict[name] = param.grad.detach().flatten().cpu()
    torch.save(grad_dict, f"{save_dir}/epoch{epoch}_batch{batch_idx}.pt")

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.clone().detach()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1.0 - self.decay) * param.detach()
                )

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.clone().detach()
                param.data.copy_(self.shadow[name].data)

    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name].data)
        self.backup = {}


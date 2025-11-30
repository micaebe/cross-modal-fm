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

def nearest_labels_l2(final_states: Tensor, means: Tensor) -> Tensor:
    flat_states = final_states.flatten(start_dim=1)
    flat_means = means.flatten(start_dim=1)
    dists = torch.cdist(flat_states, flat_means, p=2)
    return torch.argmin(dists, dim=1)

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


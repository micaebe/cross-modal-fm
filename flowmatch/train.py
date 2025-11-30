import torch
from utils import nearest_labels, log_full_grads
from torchdiffeq import odeint
from label_embeddings import build_embedding_provider
from torchvision.utils import make_grid
from einops import rearrange
import numpy as np
from PIL import Image
import os
try:
    import wandb
except Exception:
    pass


# RF class from: https://github.com/cloneofsimo/minRF/blob/main/rf.py
# modified to support different source and target distributions and bidirectional training
# sample function is using ODE integration from torchdiffeq
class RF:
    def __init__(self, model, ln=False, source="noise", target="image", embedding_type="ortho", emb_std_scale=0.5, img_dim=(32, 32, 1), lambda_b=0.5, bidirectional=False):
        # source and target can be "image", "noise, or "label"
        self.model = model
        self.ln = ln
        self.source_type = source
        self.target_type = target
        self.lambda_b = lambda_b
        self.is_bidirectional = bidirectional
        self.label_embedder = build_embedding_provider(embedding_type, H=img_dim[0], W=img_dim[1], C=img_dim[2], num_classes=10, std_scale=emb_std_scale)

    def get_distribution(self, x, labels, name):
        if name not in ["source", "target"]:
            raise ValueError("name must be either 'source' or 'target'")
        dtype = self.source_type if name == "source" else self.target_type
        if dtype == "image":
            return x
        elif dtype == "noise":
            return torch.randn_like(x)
        elif dtype == "label":
            return self.label_embedder.sample(labels).to(x.device)
        else:
            raise ValueError("dtype must be either 'image', 'noise', or 'label'")

    def forward(self, x, labels, cond=None):
        b = x.size(0)
        if self.ln:
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        
        z0 = self.get_distribution(x, labels, "source")
        z1 = self.get_distribution(x, labels, "target")
        
        bwd_mask = torch.zeros((b,), dtype=torch.bool, device=x.device)
        if self.is_bidirectional:
            # False/0 for forward, True/1 for backward
            bwd_mask = torch.rand((b,)).to(x.device) > (1.0 - self.lambda_b)
            cond = bwd_mask.long()
            mask_expanded = bwd_mask.view([b, *([1] * len(x.shape[1:]))])
            # swap
            z0_new = torch.where(mask_expanded, z1, z0)
            z1_new = torch.where(mask_expanded, z0, z1)
            z0, z1 = z0_new, z1_new
        
        zt = (1 - texp) * z0 + texp * z1
        vtheta = self.model(zt, t, cond)
        
        batchwise_mse = ((z1 - z0 - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        dlist = bwd_mask.float().detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, dir, tloss) for tv, dir, tloss in zip(t, dlist, tlist)]
        return batchwise_mse.mean(), ttloss


    @torch.no_grad()
    def sample(self, z, cond=None, null_cond=None, sample_steps=50, cfg=2.0, direction="forward"):
        b = z.size(0)
        if self.is_bidirectional:
            label = 0 if direction == "forward" else 1
            cond = torch.full((b,), label, device=z.device, dtype=torch.long)
            t0, t1 = (0.0, 1.0)
        else:
            t0, t1 = (0.0, 1.0) if direction == "forward" else (1.0, 0.0)
        t_span = torch.linspace(t0, t1, sample_steps + 1, device=z.device)

        def ode_func(t, x):
            t_batch = torch.full((x.shape[0],), t.item(), device=x.device)
            vc = self.model(x, t_batch, cond)
            if null_cond is not None and cfg != 1.0:
                vu = self.model(x, t_batch, null_cond)
                vc = vu + cfg * (vc - vu)
            return vc

        traj = odeint(ode_func, z, t_span, method="euler", atol=1e-5, rtol=1e-5)
        return traj


def train_one_epoch(rf, ema, loader, optimizer, device, use_conditioning=False, log_wandb=False, log_full_gradient=False, epoch=0):
    rf.model.train()
    running = 0.0

    for i, (x, y) in enumerate(loader):
        print(i)
        x = x.to(device)
        y = y.to(device)
        cond = None

        if use_conditioning:
            cond = y

        optimizer.zero_grad()
        loss, _ = rf.forward(x, y, cond)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            rf.model.parameters(), max_norm=float('inf')
        )
        if log_full_gradient:
            if i % 10 == 0:
                log_full_grads(rf.model, epoch, i, f"source_{rf.source_type}" + "_grads")

        optimizer.step()
        ema.update(rf.model)

        running += loss.item()
        if log_wandb:
            wandb.log({"train_loss_batch": loss.item(), "grad_norm": grad_norm, "epoch": epoch})
        
        if i > 150:
            break

    return running / max(1, len(loader))


@torch.no_grad()
def evaluate(rf: RF, ema, loader, device, steps=50, use_conditioning=False, save_dir="./eval_samples", epoch=0):
    ema.apply_shadow(rf.model)
    rf.model.eval()
    all_correct_l2 = 0
    all_correct_cos = 0
    all_count = 0

    for i, (x, y) in enumerate(loader):
        print(i)
        x = x.to(device)
        y = y.to(device)

        cond = None
        if use_conditioning:
            cond = y
        
        # forward direction
        x0 = rf.get_distribution(x, y, "source")
        x1 = rf.get_distribution(x, y, "target")

        forward_traj = rf.sample(x0, cond, sample_steps=steps, direction="forward")
        backward_traj = rf.sample(x1, cond, sample_steps=steps, direction="backward")

        label_final = None
        image_final = None

        if rf.source_type == "label":
            label_final = backward_traj[-1]
        elif rf.target_type == "label":
            # final backward is used to evaluate nearest labels
            label_final = forward_traj[-1]
        if rf.source_type == "image":
            # here we do classification based on final backward
            image_final = backward_traj[-1]
        elif rf.target_type == "image":
            # here we do classification based on final forward
            image_final = forward_traj[-1]

        if image_final is not None:
            # TODO implement image-based evaluation
            pass
        if label_final is not None:
            preds_l2, preds_cos = nearest_labels(label_final, rf.label_embedder.class_means)
            all_correct_l2 += (preds_l2 == y).sum().item()
            all_correct_cos += (preds_cos == y).sum().item()


        all_count += y.numel()
        if i > 0:
            break

    def traj_to_gif(traj, y):
        seq = []
        trajs = traj[:, :25]
        trajs = trajs[:, torch.argsort(y[:25])]
        for t in trajs:
            img = (t * 0.5 + 0.5).clamp(0, 1)
            img = make_grid(img.float(), nrow=5)
            img = rearrange(img, 'c h w -> h w c').cpu().numpy()
            seq.append(Image.fromarray((img * 255).astype(np.uint8)))
        return seq
    
    fwd_gif = traj_to_gif(forward_traj, y)
    bwd_gif = traj_to_gif(backward_traj, y)
    os.makedirs(save_dir, exist_ok=True)
    name = f"{rf.source_type}_to_{rf.target_type}"
    if rf.source_type == "label" or rf.target_type == "label":
        name += f"_{rf.label_embedder.__class__.__name__}_"
    fwd_gif[0].save(f"{save_dir}/{name}forward_sample_{epoch}.gif", save_all=True, append_images=fwd_gif[1:], duration=100, loop=0)
    bwd_gif[0].save(f"{save_dir}/{name}backward_sample_{epoch}.gif", save_all=True, append_images=bwd_gif[1:], duration=100, loop=0)

    
    ema.restore(rf.model)

    acc_cos = all_correct_cos / max(1, all_count)
    acc_l2 = all_correct_l2 / max(1, all_count)
    return acc_l2, acc_cos

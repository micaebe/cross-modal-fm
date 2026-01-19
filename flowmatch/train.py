import torch
from flowmatch.rf import RF
from flowmatch.utils import nearest_labels, log_full_grads
from torchvision.utils import make_grid
from einops import rearrange
import numpy as np
from PIL import Image
import os
try:
    import wandb
except Exception:
    pass


def train_one_epoch(rf: RF, ema, loader, optimizer, device, use_conditioning=False, log_wandb=False, log_full_gradient=False, use_bf16=False, epoch=0):
    rf.model.train()
    log_bins_every = 50
    running = 0.0

    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        cond = None # adaln conditioning

        if use_conditioning:
            cond = y
        else:
            # CrossFlow style indicators for beeing able to sample unconditionally and conditionally (to enable cfg)
            to_drop = (torch.rand(y.size(0), device=y.device) < 0.1)
            y[to_drop] = torch.randint(0, rf.label_embedder.num_classes, (to_drop.sum(),), device=y.device)
            cond = to_drop.long()

        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            vtheta, target_v, bwd_mask, t = rf.forward(x, y, cond, True)

        batchwise_mse = ((target_v.float() - vtheta.float()) ** 2).mean(dim=list(range(1, len(x.shape))))
        if (i % log_bins_every == 0) and log_wandb:
            tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
            dlist = bwd_mask.float().detach().cpu().reshape(-1).tolist()
            ttloss = [(tv, dir, tloss) for tv, dir, tloss in zip(t, dlist, tlist)]
        else:
            ttloss = []
        loss = batchwise_mse.mean()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            rf.model.parameters(), max_norm=float('inf')
        )
        if log_full_gradient:
            if i % 50 == 0:
                log_full_grads(rf.model, epoch, i, f"source_{rf.source_type}" + "_grads")
            if i > 1000:
                # avoid logging too many files
                return running / max(1, len(loader))

        optimizer.step()
        ema.update(rf.model)



        running += loss.item()
        if log_wandb:
            if i % log_bins_every == 0:
                loss_per_bin_forward = {k: 0.0 for k in range(10)}
                loss_per_bin_backward = {k: 0.0 for k in range(10)}
                count_per_bin_forward = {k: 1e-6 for k in range(10)}
                count_per_bin_backward = {k: 1e-6 for k in range(10)}

                for t, d, l in ttloss:
                    idx = int(t * 10)
                    if d == 0.0:
                        loss_per_bin_forward[idx] += l
                        count_per_bin_forward[idx] += 1.0
                    else:
                        loss_per_bin_backward[idx] += l
                        count_per_bin_backward[idx] += 1.0
                
                wandb.log({
                    **{f"lossbin_{k}": loss_per_bin_forward[k] / count_per_bin_forward[k] for k in loss_per_bin_forward},
                    **{f"lossbin_bwd_{k}": loss_per_bin_backward[k] / count_per_bin_backward[k] for k in loss_per_bin_backward},
                    "batch_loss": loss.item(),
                    "grad_norm": grad_norm,
                    "epoch": epoch,
                })
            else:
                wandb.log({
                    "batch_loss": loss.item(),
                    "grad_norm": grad_norm,
                    "epoch": epoch,
                })

    return running / max(1, len(loader))





def get_final_label_and_image(rf: RF, forward_traj, backward_traj):
    label_final = None
    image_final = None
    if rf.source_type == "label":
        label_final = backward_traj[-1]
    elif rf.target_type == "label":
        label_final = forward_traj[-1]
    if rf.source_type == "image":
        image_final = backward_traj[-1]
    elif rf.target_type == "image":
        image_final = forward_traj[-1]
    return label_final, image_final

def count_correct_image_label_generations(rf, final_image, final_label, classifier, y):
    count_image_class = 0
    count_label_cos = 0
    count_label_l2 = 0
    if final_image is not None:
        if classifier is not None:
            logits = classifier(final_image)
            preds = logits.argmax(dim=1)
            count_image_class = (preds == y).sum().item()
    if final_label is not None:
        preds_l2, preds_cos = nearest_labels(final_label, rf.label_embedder.class_means)
        count_label_l2 = (preds_l2 == y).sum().item()
        count_label_cos = (preds_cos == y).sum().item()
    return count_image_class, count_label_l2, count_label_cos



@torch.no_grad()
def evaluate(rf: RF, ema, loader, device, steps=50, use_conditioning=False, save_dir="./eval_samples", classifier=None, epoch=0):
    ema.apply_shadow(rf.model)
    rf.model.eval()
    all_correct_l2 = 0
    all_correct_cos = 0
    all_correct_class = 0
    all_count = 0

    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        cond = None
        null_cond = None
        if use_conditioning:
            cond = y
            null_cond = torch.full_like(y, rf.label_embedder.num_classes)
        else:
            # null_cond is always "class" 1 and cond is class 0 (indicators from CrossFlow)
            # 0 corresponds to conditioned, 1 to unconditioned
            cond = torch.zeros_like(y).long()
            null_cond = torch.ones_like(y).long()
        
        x0 = rf.get_distribution(x, y, "source")
        x1 = rf.get_distribution(x, y, "target")

        forward_traj = rf.sample(x0, cond, null_cond, sample_steps=steps, direction="forward")
        backward_traj = rf.sample(x1, cond, null_cond, sample_steps=steps, direction="backward")

        label_final, image_final = get_final_label_and_image(rf, forward_traj, backward_traj)
        count_image_correct, count_label_l2, count_label_cos = count_correct_image_label_generations(rf, image_final, label_final, classifier, y)

        all_correct_class += count_image_correct
        all_correct_l2 += count_label_l2
        all_correct_cos += count_label_cos 
        all_count += y.numel()

        if i > 10:
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
    fwd_gif[0].save(f"{save_dir}/fwd_{epoch}.gif", save_all=True, append_images=fwd_gif[1:], duration=100, loop=0)
    bwd_gif[0].save(f"{save_dir}/bwd_{epoch}.gif", save_all=True, append_images=bwd_gif[1:], duration=100, loop=0)

    torch.save(rf.model.state_dict(), f"{save_dir}/model_ema{epoch}.pt")
    ema.restore(rf.model)
    torch.save(rf.model.state_dict(), f"{save_dir}/model_epoch{epoch}.pt")

    acc_cos = all_correct_cos / max(1, all_count)
    acc_l2 = all_correct_l2 / max(1, all_count)
    acc_class = all_correct_class / max(1, all_count)
    return acc_l2, acc_cos, acc_class

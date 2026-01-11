import os
import torch
from torchvision.utils import make_grid
from einops import rearrange
import numpy as np
from PIL import Image
from rf import RF
from utils import nearest_labels
from evaluation.evaluation_utils import get_batch_features, compute_metrics



def rf_forward_fn(rf: RF, x, y, use_bf16, device, t=None, bidi_mask=None):
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
        vtheta, target_v, bwd_mask, t = rf.forward(x, y, True, t=t, bidi_mask=bidi_mask)
    sample_mse = ((target_v.float() - vtheta.float()) ** 2).flatten(start_dim=1).mean(dim=1)
    return sample_mse, t, bwd_mask

def train_rf(rf: RF,
            ema,
            data_iterator,
            loader,
            optimizer,
            device,
            num_steps,
            logger,
            global_step,
            use_bf16=False,
            grad_accum_steps=1,
            scheduler=None):
    rf.model.train()
    running = 0.0

    optimizer.zero_grad(set_to_none=True)

    for i in range(num_steps * grad_accum_steps):
        try:
            x, y = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            x, y = next(data_iterator)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        sample_mse, _, bwd_mask = rf_forward_fn(rf, x, y, use_bf16, device)
        loss = sample_mse.mean()
        
        loss = loss / grad_accum_steps
        loss.backward()

        if (i + 1) % grad_accum_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                    rf.model.parameters(), max_norm=float(2.0)
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            optimizer.zero_grad(set_to_none=True)
            ema.update(rf.model)
            
            # we log only fine granular for each 10th stepip
            if (i + 1) % (grad_accum_steps * 10) == 0:
                logger.add_scalar("Train/Loss", loss.item() * grad_accum_steps, global_step)
                logger.add_scalar("Train/Grad_Norm", grad_norm, global_step)
            global_step += 1
        running += loss.item() * grad_accum_steps
    return running / (num_steps * grad_accum_steps), global_step



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
    dists_l2 = 0
    sims_cos = 0
    if final_image is not None:
        if classifier is not None:
            logits = classifier(final_image)
            preds = logits.argmax(dim=1)
            count_image_class = (preds == y).sum().item()
    if final_label is not None:
        preds_l2, preds_cos, dists_l2, sims_cos = nearest_labels(final_label, rf.label_embedder.class_means)
        count_label_l2 = (preds_l2 == y).sum().item()
        count_label_cos = (preds_cos == y).sum().item()
    return count_image_class, count_label_l2, count_label_cos, dists_l2, sims_cos


@torch.no_grad()
def evaluate(rf: RF, ema, loader, device, steps=50, n_batches=10, save_dir="./eval_samples", classifier=None, epoch=0, 
             fid_model=None, fid_resizer=None, fid_stats=None, real_feats=None):
    # simple evaluation during training
    ema.apply_shadow(rf.model)
    rf.model.eval()
    all_correct_l2 = 0
    all_correct_cos = 0
    all_correct_class = 0
    all_count = 0
    sum_l2_dist = 0.0
    sum_cos_sim = 0.0
    
    all_gen_feats = []

    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        x = x.to(device)
        y = y.to(device)

        if rf.use_conditioning:
            cond = y
            null_cond = None #torch.full_like(y, rf.label_embedder.num_classes)
        else:
            # 0 corresponds to conditioned, 1 to unconditioned
            if rf.cfg_dropout_prob < 1.0:
                # -> do conditional generation
                cond = torch.zeros_like(y).long()
            else:
                # in case cfg dropout = 1 -> we are train a unconditional model
                cond = torch.ones_like(y).long()
            null_cond = None #torch.ones_like(y).long()
        
        x0 = rf.get_endpoints(x, y, "source")
        x1 = rf.get_endpoints(x, y, "target")

        forward_traj = rf.sample(x0, cond, null_cond, sample_steps=steps, direction="forward", cfg=1.0)
        backward_traj = rf.sample(x1, cond, null_cond, sample_steps=steps, direction="backward", cfg=1.0)

        label_final, image_final = get_final_label_and_image(rf, forward_traj, backward_traj)
        count_image_correct, count_label_l2, count_label_cos, dists_l2, sims_cos = count_correct_image_label_generations(rf, image_final, label_final, classifier, y)

        if fid_model is not None and image_final is not None:
            feats = get_batch_features(image_final, fid_model, fid_resizer, device)
            all_gen_feats.append(feats)

        all_correct_class += count_image_correct
        all_correct_l2 += count_label_l2
        all_correct_cos += count_label_cos 
        sum_l2_dist += dists_l2[:, y].diag().sum().item()
        sum_cos_sim += sims_cos[:, y].diag().sum().item()
        all_count += y.numel()

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

    ema.restore(rf.model)

    acc_cos = all_correct_cos / max(1, all_count)
    acc_l2 = all_correct_l2 / max(1, all_count)
    acc_class = all_correct_class / max(1, all_count)
    mean_l2 = sum_l2_dist / max(1, all_count)
    mean_cos = sum_cos_sim / max(1, all_count)
    
    metrics = {
        "acc_l2": acc_l2,
        "acc_cos": acc_cos,
        "acc_class": acc_class,
        "mean_l2": mean_l2,
        "mean_cos": mean_cos
    }
    
    if all_gen_feats:
        gen_feats = np.concatenate(all_gen_feats, axis=0)
        fid_prdc_metrics = compute_metrics(gen_feats, real_feats, fid_stats)
        metrics.update(fid_prdc_metrics)
        
    return metrics

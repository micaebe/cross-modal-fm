import torch
import numpy as np
from einops import rearrange
from PIL import Image
from ..rf import RF
from .utils import (
    traj_to_gif,
    to_image_space,
    get_batch_features,
    compute_metrics,
    get_final_label_and_image,
    count_correct_image_label_generations
)
import os



@torch.no_grad()
def evaluate(rf: RF, loader, device, steps=50, cfg_scale=1.0, n_batches=10, num_classes=10, save_dir="./eval_samples", classifier=None, epoch=0, 
             fid_model=None, fid_resizer=None, fid_stats=None, real_feats=None, vae=None, save_gif=True, save_samples=False):
    """
    Evaluate the RF model on the given data loader.

    Args:
        rf: The RF wrapper.
        loader: The data loader.
        device: The device.
        steps: The number of integration steps to use.
        cfg_scale: The classifier-free-guidance scale to use.
        n_batches: The number of batches to evaluate.
        num_classes: The number of classes in the dataset.
        save_dir: The directory to save the evaluation samples.
        classifier: Optional a pre-trained classifier model.
        epoch: The current epoch.
        fid_model: Optional cleanfid inception model.
        fid_resizer: Optional cleanfid resizer.
        fid_stats: Optional FID statistics of the reference images.
        real_feats: Optional Inception features of the reference images.
        vae: Optional, the VAE model if images are in latent space.
        save_gif: Whether to save gifs of the generated samples.
        save_samples: Whether to save the generated samples.
    """
    if vae:
        # we load vae only during evaluation onto gpu
        vae.to(device)
    eval_generator = torch.Generator(device=device)
    rf.model.eval()
    
    metrics_accum = {
        "acc_l2": 0,
        "acc_cos": 0,
        "acc_class": 0,
        "mean_l2": 0.0,
        "mean_cos": 0.0
    }
    all_count = 0
    all_gen_feats = []
    all_labels = []

    def get_conds(y):
        if rf.use_conditioning:
            cond = y
            null_cond = None if cfg_scale == 1.0 else torch.full_like(y, num_classes)
        else:
            # 0 corresponds to conditioned, 1 to unconditioned
            if rf.cls_dropout_prob < 1.0:
                # -> do conditional generation
                cond = torch.zeros_like(y).long()
            else:
                # in case cfg dropout = 1 -> we are train a unconditional model
                cond = torch.ones_like(y).long()
            null_cond = None if cfg_scale == 1.0 else torch.ones_like(y).long()
        return cond, null_cond

    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        cond, null_cond = get_conds(y)

        # using different seed per batch (per sample would be probably better)
        eval_generator.manual_seed(42 + i)
        eval_noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=eval_generator)

        # resolve endpoints (source -> target; t=0 -> t=1)
        x0 = rf.resolve_endpoints(x, y, "source", eval_noise)
        x1 = rf.resolve_endpoints(x, y, "target", eval_noise)

        # integrate forward (t=0 -> t=1) and backwards (t=1 -> t=0)
        forward_traj = rf.sample(x0, cond, null_cond, sample_steps=steps, direction="forward", cfg=cfg_scale)
        backward_traj = rf.sample(x1, cond, null_cond, sample_steps=steps, direction="backward", cfg=cfg_scale)

        # resolve generated endpoints
        label_final, image_final = get_final_label_and_image(rf, forward_traj, backward_traj)
        count_image_correct, count_label_l2, count_label_cos, dists_l2, sims_cos = count_correct_image_label_generations(rf, image_final, label_final, classifier, y)

        if fid_model is not None and image_final is not None:
            if vae:
                image_final = to_image_space(image_final, vae)
            feats = get_batch_features(image_final, fid_model, fid_resizer, device, None)
            all_gen_feats.append(feats)
            all_labels.append(y.cpu().numpy())

        batch_metrics = {
            "acc_class": count_image_correct,
            "acc_l2": count_label_l2,
            "acc_cos": count_label_cos,
            "mean_l2": dists_l2[:, y].diag().sum().item(),
            "mean_cos": sims_cos[:, y].diag().sum().item()
        }

        for k, v in batch_metrics.items():
            metrics_accum[k] += v

        all_count += y.numel()

        if i+1 >= n_batches:
            break

    if save_gif:
        os.makedirs(save_dir, exist_ok=True)
        traj_to_gif(forward_traj, y, f"{save_dir}/fwd_{epoch}.gif")
        traj_to_gif(backward_traj, y, f"{save_dir}/bwd_{epoch}.gif")
    if save_samples:
        os.makedirs(save_dir, exist_ok=True)
        for class_idx in range(num_classes):
            class_dir = os.path.join(save_dir, str(class_idx))
            os.makedirs(class_dir, exist_ok=True)
            idxs = (y == class_idx).nonzero(as_tuple=True)[0]
            for idx in idxs:
                lbls = label_final[idx]
                imgs = image_final[idx]
                lbls = (lbls * 0.5 + 0.5).clamp(0, 1) * 255
                imgs = (imgs * 0.5 + 0.5).clamp(0, 1) * 255
                imgs = rearrange(imgs, 'c h w -> h w c').cpu().numpy()
                lbls = rearrange(lbls, 'c h w -> h w c').cpu().numpy()
                Image.fromarray(lbls.astype(np.uint8)).save(os.path.join(class_dir, f"label_{epoch}_{idx}.png"))
                Image.fromarray(imgs.astype(np.uint8)).save(os.path.join(class_dir, f"image_{epoch}_{idx}.png"))

    metrics = {k: v / max(1, all_count) for k, v in metrics_accum.items()}

    if all_gen_feats:
        gen_feats = np.concatenate(all_gen_feats, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        fid_prdc_metrics = compute_metrics(gen_feats, real_feats, fid_stats, all_labels)
        metrics.update(fid_prdc_metrics)

    if vae:
        vae.to("cpu")
    return metrics
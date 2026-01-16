import torch
import numpy as np
import prdc
from cleanfid import fid
from cleanfid.resize import build_resizer
from torchvision.utils import make_grid
from einops import rearrange
from PIL import Image
from rf import RF
from utils import nearest_labels
from pathlib import Path
import os



def resize_and_quantize(batch_tensor, resizer):
    batch_denorm = (batch_tensor + 1) / 2 * 255
    batch_denorm = batch_denorm.clamp(0, 255).to(torch.uint8)
    if batch_denorm.shape[1] == 1:
        # in case of MNIST, we need to repeat the channels
        # Inception features probably not the best for MNIST
        # logging it anyways
        batch_denorm = batch_denorm.repeat(1, 3, 1, 1)
    batch_np = batch_denorm.permute(0, 2, 3, 1).cpu().numpy()
    
    resized_list = []
    for i in range(batch_np.shape[0]):
        img_np = batch_np[i]
        resized_np = resizer(img_np)
        resized_list.append(resized_np)
        
    resized_batch_np = np.stack(resized_list)
    resized_batch = torch.from_numpy(resized_batch_np).permute(0, 3, 1, 2).float()
    return resized_batch

def get_batch_features(batch_tensor, fid_model, fid_resizer, device, vae=None):
    if vae:
        batch_tensor = to_image_space(batch_tensor, vae)
    resized_batch = resize_and_quantize(batch_tensor, fid_resizer)
    feats = fid.get_batch_features(resized_batch, fid_model, device=device)
    return feats

def compute_metrics(gen_feats, real_feats, fid_stats=None):
    metrics = {}
    mu_gen = np.mean(gen_feats, axis=0)
    sigma_gen = np.cov(gen_feats, rowvar=False)
    if fid_stats is not None:
        mu_ref, sigma_ref = fid_stats
        fid_score = fid.frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)
        metrics['fid'] = fid_score
    if real_feats is not None:
        prdc_res = prdc.compute_prdc(real_features=real_feats, fake_features=gen_feats, nearest_k=5)
        metrics.update(prdc_res)
        # if fid was not computed via fid stats, we compute it using the real_feats
        if fid_stats is None:
            mu_real = np.mean(real_feats, axis=0)
            sigma_real = np.cov(real_feats, rowvar=False)
            fid_score = fid.frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
            metrics['fid'] = fid_score
    return metrics

def get_real_features_for_dataset(loader, fid_model, fid_resizer, device, max_batches=None, vae=None):
    all_real_feats = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device)
            feats = get_batch_features(x, fid_model, fid_resizer, device, vae)
            all_real_feats.append(feats)
    
    real_feats = np.concatenate(all_real_feats, axis=0)
    return real_feats

def get_fid_components(dataset_name, device, fid_ref_dir=None):
    fid_model = None
    fid_resizer = None
    fid_stats = None
    
    if dataset_name in ["cifar", "imagenet100", "mnist"]:
         fid_model = fid.build_feature_extractor(mode="clean", device=torch.device(device))
         fid_resizer = build_resizer(mode="clean")

    if dataset_name == "cifar":
        fid_stats = fid.get_reference_statistics("cifar10", 32, split="test")
    elif dataset_name == "mnist":
        pass
    elif dataset_name == "imagenet100":
        if fid_ref_dir:
            mu, sigma = fid.get_folder_features(fid_ref_dir, model=fid_model, num_workers=0, batch_size=128, device=torch.device(device), mode="clean")
            fid_stats = (mu, sigma)

    # if no fid_stats are provided, we calcuate them at the beginning of the training (see compute_metrics function above)
    # probably a bit inefficient
    return fid_model, fid_resizer, fid_stats


def get_vae(cfg):
    if cfg.dataset.name != "imagenet100":
        return None
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to("cpu").eval()
    return vae

@torch.no_grad()
def to_image_space(batch_tensor, vae):
    x = vae.decode(batch_tensor / 0.13025).sample
    return x


def traj_to_gif(traj, y, gif_path):
    seq = []
    trajs = traj[:, :25]
    trajs = trajs[:, torch.argsort(y[:25])]
    for t in trajs:
        img = (t * 0.5 + 0.5).clamp(0, 1)
        img = make_grid(img.float(), nrow=5)
        img = rearrange(img, 'c h w -> h w c').cpu().numpy()
        seq.append(Image.fromarray((img * 255).astype(np.uint8)))
    seq[0].save(gif_path, save_all=True, append_images=seq[1:], duration=100, loop=0)
    return seq


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
def evaluate(rf: RF, loader, device, steps=50, cfg_scale=1.0, n_batches=10, num_classes=10, save_dir: str | Path ="./eval_samples", classifier=None, epoch=0, 
             fid_model=None, fid_resizer=None, fid_stats=None, real_feats=None, vae=None, save_gif=True, save_samples=True):
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
        fid_prdc_metrics = compute_metrics(gen_feats, real_feats, fid_stats)
        metrics.update(fid_prdc_metrics)

    if vae:
        vae.to("cpu")
    return metrics
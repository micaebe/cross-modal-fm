import torch.nn.functional as F
import torch
import numpy as np
import prdc
from cleanfid import fid
from cleanfid.resize import build_resizer
from torchvision.utils import make_grid
from einops import rearrange
from PIL import Image


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


def get_final_label_and_image(rf, forward_traj, backward_traj):
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
import torch
import numpy as np
import prdc
from cleanfid import fid
from cleanfid.resize import build_resizer


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

def get_batch_features(batch_tensor, fid_model, fid_resizer, device):
    resized_batch = resize_and_quantize(batch_tensor, fid_resizer)
    feats = fid.get_batch_features(resized_batch, fid_model, device=device)
    return feats

def compute_metrics(gen_feats, real_feats, fid_stats=None):
    metrics = {}
    if fid_stats is not None:
        mu_gen = np.mean(gen_feats, axis=0)
        sigma_gen = np.cov(gen_feats, rowvar=False)
        mu_ref, sigma_ref = fid_stats
        fid_score = fid.frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)
        metrics['fid'] = fid_score
    if real_feats is not None:
        prdc_res = prdc.compute_prdc(real_features=real_feats, fake_features=gen_feats, nearest_k=5)
        metrics.update(prdc_res)
    return metrics

def get_real_features_for_dataset(loader, fid_model, fid_resizer, device, max_batches=None):
    all_real_feats = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device)
            feats = get_batch_features(x, fid_model, fid_resizer, device)
            all_real_feats.append(feats)
    
    real_feats = np.concatenate(all_real_feats, axis=0)
    return real_feats

def get_fid_components(dataset_name, device, fid_ref_dir=None):
    fid_model = None
    fid_resizer = None
    fid_stats = None
    
    if dataset_name in ["cifar", "imagenet", "mnist"]:
         fid_model = fid.build_feature_extractor(mode="clean", device=torch.device(device))
         fid_resizer = build_resizer(mode="clean")

    if dataset_name == "cifar":
        fid_stats = fid.get_reference_statistics("cifar10", 32, split="test")
    elif dataset_name == "mnist":
        pass
    elif dataset_name == "imagenet":
        if fid_ref_dir:
            mu, sigma = fid.get_folder_features(fid_ref_dir, model=fid_model, num_workers=0, batch_size=128, device=torch.device(device), mode="clean")
            fid_stats = (mu, sigma)
    return fid_model, fid_resizer, fid_stats

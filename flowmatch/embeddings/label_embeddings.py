import torch
import torch.nn as nn
from torchvision.transforms.functional import gaussian_blur
from einops import rearrange, repeat
import itertools
import math
import os

class BaseEmbedding(nn.Module):
    def __init__(self, H, W, C, num_classes, std_scale):
        super().__init__()
        self.H, self.W, self.C = H, W, C
        self.num_classes = num_classes
        self.std_scale = std_scale

    @torch.no_grad()
    def forward(self, labels, sample: bool = True):
        batch_means = self.class_means[labels]
        batch_stds = self.class_stds[labels]

        if sample:
            noise = torch.randn_like(batch_stds, device=labels.device)
            return batch_means + batch_stds * noise
        
        return batch_means
    
    def _setup(self, means, stds):
        self.register_buffer("class_means", means)
        self.register_buffer("class_stds", stds)


class GrayScaleEmbedding(BaseEmbedding):
    def __init__(self, H, W, C, num_classes, std_scale):
        super().__init__(H, W, C, num_classes, std_scale)
        vals = torch.linspace(-1.0, 1.0, steps=num_classes)
        means = repeat(vals, 'n -> n c h w', c=C, h=H, w=W)
        stds = torch.full((num_classes, C, H, W), std_scale)
        self._setup(means, stds)


class RectangleEmbedding(BaseEmbedding):
    def __init__(self, H, W, C, num_classes, std_scale, blur_sigma, codes_per_cell, patch_size=None):
        """
        Args:
            patch_size: Optional tuple (patch_h, patch_w) or int for square patches. 
                        If None, patches fill the grid cells (original behavior).
        """
        super().__init__(H, W, C, num_classes, std_scale)
        
        codes_map = {
            1: [1.0],
            2: [-1.0, 1.0],
        }
        if codes_per_cell in codes_map:
            code_list = [[v] * C for v in codes_map[codes_per_cell]]
        else:
            code_list = [list(x) for x in itertools. product([-1.0, 1.0], repeat=C)]

        code_list = [torch.tensor(c, dtype=torch.float32) for c in code_list]
        codes = torch.stack(code_list)
        num_codes = len(codes)

        num_cells = math.ceil(num_classes / num_codes)
        cols = math.ceil(math.sqrt(num_cells))
        rows = math.ceil(num_cells / cols)
        
        cell_h = H / rows
        cell_w = W / cols
        
        if patch_size is None:
            patch_h = int(cell_h)
            patch_w = int(cell_w)
        elif isinstance(patch_size, int):
            patch_h = patch_w = patch_size
        else:
            patch_h, patch_w = patch_size

        if num_codes * num_cells > num_classes:
            codes = codes[:num_classes // num_cells]
            num_codes = len(codes)

        print(f"Rect. Emb: Grid: {rows}x{cols} | Cell:  {cell_h:.2f}x{cell_w:.2f} | Patch: {patch_h}x{patch_w} | Num Codes: {num_codes}")

        means = torch.zeros(num_classes, C, H, W)
        stds = torch.full((num_classes, C, H, W), std_scale)

        for i in range(num_classes):
            cell_idx = i // num_codes
            code_idx = i % num_codes
            
            r, c = divmod(cell_idx, cols)

            center_y = int((r + 0.5) * cell_h)
            center_x = int((c + 0.5) * cell_w)

            y0 = max(0, center_y - patch_h // 2)
            x0 = max(0, center_x - patch_w // 2)
            y1 = min(H, y0 + patch_h)
            x1 = min(W, x0 + patch_w)

            means[i, :, y0:y1, x0:x1] = codes[code_idx].view(C, 1, 1)

        if blur_sigma > 0:
            means = gaussian_blur(means, kernel_size=7, sigma=blur_sigma)
            max_val = means.abs().amax(dim=(1, 2, 3), keepdim=True)
            means = means / (max_val + 1e-8)

        self._setup(means, stds)


class SmoothRandom(BaseEmbedding):
    def __init__(self, H, W, C, num_classes, std_scale, low_res_dim=4, cache_dir="./embeddings_cache"):
        super().__init__(H, W, C, num_classes, std_scale)
        self.low_res_dim = low_res_dim
        self.cache_dir = cache_dir
        means, stds = self._load_or_create()
        self._setup(means, stds)

    def _get_config_name(self):
        return f"smoothrand_C{self.num_classes}_H{self.H}W{self.W}C{self.C}_S{self.std_scale}_L{self.low_res_dim}.pt"

    def _load_or_create(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        file_path = os.path.join(self.cache_dir, self._get_config_name())

        if os.path.exists(file_path):
            data = torch.load(file_path, map_location="cpu")
            return data["means"], data["stds"]
        else:
            print(f"LR Emb: Generating and caching: {file_path}")
            return self._generate_and_save(file_path)

    def _generate_and_save(self, file_path):
        g = torch.Generator()
        g.manual_seed(42)

        low_res_means = torch.randn(
            self.num_classes, self.C, self.low_res_dim, self.low_res_dim, 
            generator=g
        ).clamp(-3.0, 3.0)
        mins = low_res_means.amin(dim=(1, 2, 3), keepdim=True)
        maxs = low_res_means.amax(dim=(1, 2, 3), keepdim=True)
        low_res_means = (low_res_means - mins) / (maxs - mins) * 2 - 1
        means = torch.nn.functional.interpolate(
            low_res_means,
            size=(self.H, self.W),
            mode='bilinear',
            align_corners=False
        )
        stds = torch.full((self.num_classes, self.C, self.H, self.W), self.std_scale)

        torch.save({"means": means, "stds": stds}, file_path)
        return means, stds


class ClipEmbedding(BaseEmbedding):
    def __init__(self, H, W, C, num_classes, std_scale, dataset_name="mnist",
                 model_name="openai/clip-vit-base-patch32", clip_device="cpu"):
        super().__init__(H, W, C, num_classes, std_scale)
        from transformers import CLIPTokenizer, CLIPTextModel

        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        model = CLIPTextModel.from_pretrained(model_name).to(clip_device)
        g = torch.Generator().manual_seed(42)

        if dataset_name == "mnist":
            label_texts = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
        elif dataset_name == "cifar":
            label_texts = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
        else:
            raise ValueError("Only mnist and cifar10 datasets are currently supported for CLIP embeddings")

        labels_txt = label_texts[:num_classes]
        inputs = tokenizer(labels_txt, padding=True, return_tensors="pt").to(clip_device)
        with torch.no_grad():
            outputs = model(**inputs)
        text_embeds = outputs.pooler_output

        D = H * W * C
        A = torch.randn(D, 512, device=clip_device, generator=g)
        Q, _ = torch.linalg.qr(A, mode='reduced')
        P = Q[:, :512]
        
        mapped = text_embeds @ P.T  # (N, D)

        means = rearrange(mapped, 'n (c h w) -> n c h w', c=C, h=H, w=W)
        stds = torch.full((num_classes, C, H, W), std_scale, device=clip_device)

        self._setup(means, stds)


import torch
import torch.nn as nn
import torch.nn.functional as F
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
            noise = torch.randn_like(batch_stds)
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
    def __init__(self, H, W, C, num_classes, std_scale, blur_sigma, codes_per_cell):
        super().__init__(H, W, C, num_classes, std_scale)
        codes_map = {
            1: [1.0],
            2: [-1.0, 1.0],
            3: [-1.0, 0.0, 1.0],
            4: [-1.0, -0.33, 0.33, 1.0],
            5: [-1.0, -0.5, 0.0, 0.5, 1.0]
        }
        code_list = list(itertools.product(codes_map[codes_per_cell], repeat=C))
        code_list = [torch.tensor(c, dtype=torch.float32) for c in code_list if sum(c) > 0]
        
        code_list.sort(key=lambda x: x.sum())
        codes = torch.stack(code_list)
        num_codes = len(codes)

        num_cells = math.ceil(num_classes / num_codes)
        
        cols = math.ceil(math.sqrt(num_cells))
        rows = math.ceil(num_cells / cols)
        
        cell_h = H // rows
        cell_w = W // cols
        
        rh = max(1, int(cell_h * 0.9))
        rw = max(1, int(cell_w * 0.9))
        
        print(f"Embedding Config: {rows}x{cols} Grid ({cell_h}x{cell_w} cells). "
              f"Codes per cell: {num_codes}. Patch Size: {rh}x{rw}")

        means = torch.zeros(num_classes, C, H, W)
        stds = torch.full((num_classes, C, H, W), std_scale)

        for i in range(num_classes):
            cell_idx = i // num_codes
            code_idx = i % num_codes
            
            r = cell_idx // cols
            c = cell_idx % cols
            center_y = r * cell_h + cell_h // 2
            center_x = c * cell_w + cell_w // 2
            
            y0 = int(center_y - rh / 2)
            x0 = int(center_x - rw / 2)
            y0 = max(0, min(y0, H - rh))
            x0 = max(0, min(x0, W - rw))
            
            current_code = codes[code_idx].view(C, 1, 1)
            means[i, :, y0 : y0 + rh, x0 : x0 + rw] = current_code

        if blur_sigma > 0:
            means = gaussian_blur(means, kernel_size=5, sigma=blur_sigma)
            means /= means.amax(dim=(1, 2, 3), keepdim=True)

        self._setup(means, stds)


class LowRankEmbedding(BaseEmbedding):
    def __init__(self, H, W, C, num_classes, std_scale, low_res_dim=4, cache_dir="../embeddings_cache"):
        super().__init__(H, W, C, num_classes, std_scale)
        self.low_res_dim = low_res_dim
        self.cache_dir = cache_dir
        self.seed = 42
        means, stds = self._load_or_create()
        self._setup(means, stds)

    def _get_config_name(self):
        return f"lowrank_C{self.num_classes}_H{self.H}W{self.W}C{self.C}_S{self.std_scale}_L{self.low_res_dim}_seed{self.seed}.pt"

    def _load_or_create(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        file_path = os.path.join(self.cache_dir, self._get_config_name())

        if os.path.exists(file_path):
            data = torch.load(file_path, map_location="cpu")
            return data["means"], data["stds"]
        else:
            print(f"Generating and caching: {file_path}")
            return self._generate_and_save(file_path)

    def _generate_and_save(self, file_path):
        g = torch.Generator()
        g.manual_seed(self.seed)

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
    def __init__(self, H, W, C, num_classes, std_scale, 
                 model_name="openai/clip-vit-base-patch32", clip_device="cpu"):
        super().__init__(H, W, C, num_classes, std_scale)
        from transformers import CLIPTokenizer, CLIPTextModel

        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        model = CLIPTextModel.from_pretrained(model_name).to(clip_device)
        
        defaults = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
        if num_classes > len(defaults):
            raise ValueError(f"only 10 classes supported, got {num_classes}")
        labels_txt = defaults[:num_classes]

        inputs = tokenizer(labels_txt, padding=True, return_tensors="pt").to(clip_device)
        with torch.no_grad():
            outputs = model(**inputs)
        text_embeds = outputs.pooler_output  # (N, 512)

        D = H * W * C
        A = torch.randn(D, 512, device=clip_device)
        Q, _ = torch.linalg.qr(A, mode='reduced')
        P = Q[:, :512]
        
        mapped = text_embeds @ P.T  # (N, D)

        means = rearrange(mapped, 'n (c h w) -> n c h w', c=C, h=H, w=W)
        stds = torch.full((num_classes, C, H, W), std_scale, device=clip_device)

        self._setup(means, stds)


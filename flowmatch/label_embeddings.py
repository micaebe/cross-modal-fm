import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import os

class BaseEmbedding(nn.Module):
    def __init__(self, H, W, C, num_classes, std_scale):
        super().__init__()
        self.H, self.W, self.C = H, W, C
        self.num_classes = num_classes
        self.std_scale = std_scale

    def sample(self, labels: torch.Tensor, sample: bool = True) -> torch.Tensor:
        batch_means = self.class_means[labels]
        batch_stds = self.class_stds[labels]

        if sample:
            noise = torch.randn_like(batch_stds)
            return batch_means + batch_stds * noise
        
        return batch_means


class GrayScaleEmbedding(BaseEmbedding):
    def __init__(self, H=28, W=28, C=1, num_classes=10, std_scale=0.3):
        super().__init__(H, W, C, num_classes, std_scale)

        vals = torch.linspace(-1.0, 1.0, steps=num_classes)
        
        means = repeat(vals, 'n -> n c h w', c=C, h=H, w=W)
        stds = torch.full((num_classes, C, H, W), std_scale)

        self.register_buffer("class_means", means)
        self.register_buffer("class_stds", stds)


class RectangleEmbedding(BaseEmbedding):
    def __init__(self, H=28, W=28, C=1, num_classes=10, std_scale=0.3, rect_size=(8, 8)):
        super().__init__(H, W, C, num_classes, std_scale)
        means = torch.zeros(num_classes, C, H, W)
        stds = torch.full((num_classes, C, H, W), std_scale)
        
        cols = math.ceil(math.sqrt(num_classes))
        rows = math.ceil(num_classes / cols)
        
        cell_h = H // rows
        cell_w = W // cols
        rh, rw = rect_size

        for i in range(num_classes):
            r = i // cols
            c = i % cols
            
            center_y = r * cell_h + cell_h // 2
            center_x = c * cell_w + cell_w // 2
            
            y0 = int(center_y - rh / 2)
            x0 = int(center_x - rw / 2)
            y0 = max(0, min(y0, H - rh))
            x0 = max(0, min(x0, W - rw))
            
            means[i, :, y0 : y0 + rh, x0 : x0 + rw] = 1.0

        self.register_buffer("class_means", means)
        self.register_buffer("class_stds", stds)


class OrthoEmbedding(BaseEmbedding):
    def __init__(self, H=28, W=28, C=1, num_classes=10, std_scale=0.5, 
                 cache_dir="./embeddings_cache", seed=42):
        super().__init__(H, W, C, num_classes, std_scale)
        self.cache_dir = cache_dir
        self.seed = seed
        self.dim_total = H * W * C
        
        if self.dim_total < num_classes:
            raise ValueError(f"Dim {self.dim_total} too small for {num_classes} orthogonal vectors.")

        means, stds = self._load_or_create()

        self.register_buffer("class_means", means)
        self.register_buffer("class_stds", stds)

    def _get_config_name(self):
        return f"ortho_C{self.num_classes}_H{self.H}W{self.W}C{self.C}_S{self.std_scale}_seed{self.seed}.pt"

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

        random_matrix = torch.randn(self.dim_total, self.num_classes, generator=g)
        Q, _ = torch.linalg.qr(random_matrix, mode='reduced')

        mean_radius = (self.dim_total * (1 - self.std_scale**2)) ** 0.5
        means_flat = Q * mean_radius

        means = rearrange(means_flat, '(c h w) n -> n c h w', c=self.C, h=self.H, w=self.W)
        stds = torch.full((self.num_classes, self.C, self.H, self.W), self.std_scale)

        torch.save({"means": means, "stds": stds}, file_path)
        return means, stds


class ClipEmbedding(BaseEmbedding):
    def __init__(self, H=28, W=28, C=1, num_classes=10, std_scale=0.1, 
                 model_name="openai/clip-vit-base-patch32", device="cpu"):
        super().__init__(H, W, C, num_classes, std_scale)
        try:
            from transformers import CLIPTokenizer, CLIPTextModel
        except ImportError as e:
            raise RuntimeError("transformers not installed.") from e

        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        model = CLIPTextModel.from_pretrained(model_name).to(device)
        
        defaults = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
        if num_classes > len(defaults):
            raise ValueError(f"only 10 classes supported, got {num_classes}")
        labels_txt = defaults[:num_classes]

        inputs = tokenizer(labels_txt, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        text_embeds = outputs.pooler_output  # (N, 512)

        D = H * W * C
        A = torch.randn(D, 512, device=device)
        Q, _ = torch.linalg.qr(A, mode='reduced')
        P = Q[:, :512]
        
        mapped = text_embeds @ P.T  # (N, D)

        means = rearrange(mapped, 'n (c h w) -> n c h w', c=C, h=H, w=W)
        stds = torch.full((num_classes, C, H, W), std_scale, device=device)

        self.register_buffer("class_means", means)
        self.register_buffer("class_stds", stds)



import inspect

EMBEDDINGS = {
    "grayscale": GrayScaleEmbedding,
    "rectangle": RectangleEmbedding,
    "ortho": OrthoEmbedding,
    "clip": ClipEmbedding,
}


def build_embedding_provider(name: str, **kwargs) -> BaseEmbedding:
    if name not in EMBEDDINGS:
        raise ValueError(f"Unknown embedding provider: {name}")
    emb = EMBEDDINGS[name]
    sig = inspect.signature(emb)
    filtered_args = {
        k: v for k, v in kwargs.items() 
        if k in sig.parameters
    }
    return emb(**filtered_args)
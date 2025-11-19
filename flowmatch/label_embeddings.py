import numpy as np
import torch
from torch import Tensor

class LabelEmbedding:
    def __init__(self, H=28, W=28, C=1, num_classes=10, device="cpu"):
        self.H, self.W, self.C = H, W, C
        self.num_classes = num_classes
        self.device = device

    def means(self) -> Tensor:
        raise NotImplementedError

    def stds(self) -> Tensor:
        raise NotImplementedError

    def sample(self, labels: Tensor, sample: bool = True) -> Tensor:
        mu = self.means()
        sig = self.stds()
        out = []
        for l in labels:
            m = mu[l]
            s = sig[l]
            if sample:
                x = m + s * torch.randn_like(s)
            else:
                x = m
            out.append(x)
        out = torch.stack(out, dim=0)
        return out.permute(0, 3, 1, 2).contiguous()

class ScalarLevelsEmbedding(LabelEmbedding):
    def __init__(self, H=28, W=28, C=1, num_classes=10, std_scale=0.3, device="cpu"):
        super().__init__(H, W, C, num_classes, device)
        vals = torch.linspace(-1.0, 1.0, steps=num_classes)
        means = []
        stds = []
        for v in vals:
            means.append(torch.ones(H, W, C) * v)
            stds.append(torch.ones(H, W, C) * std_scale)
        self._means = torch.stack(means, dim=0)
        self._stds = torch.stack(stds, dim=0)

    def means(self):
        return self._means.to(self.device)

    def stds(self):
        return self._stds.to(self.device)

class ClipEmbedding(LabelEmbedding):
    def __init__(self, H=28, W=28, C=1, num_classes=10, std_scale=0.1, device="cpu", model_name="openai/clip-vit-base-patch32"):
        super().__init__(H, W, C, num_classes, device)
        try:
            from transformers import CLIPTokenizer, CLIPTextModel
        except Exception as e:
            raise RuntimeError("transformers not installed or CLIP model not available. Install transformers to use ClipTextProjectionProvider.") from e

        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        model = CLIPTextModel.from_pretrained(model_name).to(device)
        labels_txt = ["Zero","One","Two","Three","Four","Five","Six","Seven","Eight","Nine"][:num_classes]
        inputs = tokenizer(labels_txt, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        text_embeds = outputs.pooler_output  # (K, 512)

        D = H * W * C
        # random projection from 512 -> D
        A = torch.randn(D, 512, device=device)
        Q, _ = torch.linalg.qr(A)
        P = Q[:, :512]
        mapped = text_embeds @ P.T
        means = mapped.view(num_classes, H, W, C)

        self._means = means.detach().to("cpu")
        self._stds = torch.ones_like(self._means) * std_scale

    def means(self):
        return self._means.to(self.device)

    def stds(self):
        return self._stds.to(self.device)

class CircleEmbedding(LabelEmbedding):
    def __init__(self, H=28, W=28, C=1, num_classes=10, radius=7.5, std_scale=0.85, device="cpu", seed=42):
        super().__init__(H, W, C, num_classes, device)
        np.random.seed(seed)
        torch.manual_seed(seed)

        D = H * W * C
        angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
        latents2d = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

        A = np.random.normal(size=(D, 2))
        Q, _ = np.linalg.qr(A)
        proj = Q[:, :2]
        means_flat = (proj @ latents2d.T).T
        means = means_flat.reshape(num_classes, H, W, C)
        stds = np.abs(std_scale * (0.8 + 0.2 * np.random.random(size=(num_classes, H, W, C))))

        self._means = torch.from_numpy(means).float()
        self._stds = torch.from_numpy(stds).float()

    def means(self):
        return self._means.to(self.device)

    def stds(self):
        return self._stds.to(self.device)
    
class RandomEmbedding(LabelEmbedding):
    def __init__(self, H=28, W=28, C=1, num_classes=10, std=0.1, device="cpu", seed=42):
        super().__init__(H, W, C, num_classes, device)
        np.random.seed(seed)
        torch.manual_seed(seed)

        D = H * W * C
        means = []
        for _ in range(num_classes):
            v = torch.randn(D)
            v = v / torch.norm(v)
            means.append(v.view(H, W, C))
        self._means = torch.stack(means, dim=0)
        self._stds = torch.ones_like(self._means) * std
    
    def means(self):
        return self._means.to(self.device)
    
    def stds(self):
        return self._stds.to(self.device)


def make_embedding_provider(name: str, device="cpu", **kwargs) -> LabelEmbedding:
    name = name.lower()
    if name == "scalar":
        return ScalarLevelsEmbedding(device=device, **kwargs)
    if name == "clip":
        return ClipEmbedding(device=device, **kwargs)
    if name == "circle":
        return CircleEmbedding(device=device, **kwargs)
    if name == "random":
        return RandomEmbedding(device=device, **kwargs)
    raise ValueError(f"Unknown embedding: {name}")


if __name__ == "__main__":
    provider = make_embedding_provider("scalar", num_classes=10, H=28, W=28, C=1, device="cpu")
    labels = torch.arange(10)
    samples = provider.sample(labels, sample=True)
    print("Sampled embeddings shape:", samples.shape)  # (10, C, H, W)
    print("Means of samples: ", samples.mean((1, 2, 3)))
    print("Std of samples: ", samples.std((1, 2, 3)))
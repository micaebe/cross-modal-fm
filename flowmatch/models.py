import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinTimeEmbedding(nn.Module):
    def __init__(self, emb_dim: int, max_period: float = 10000.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_period = max_period

        half = emb_dim // 2
        denom = max(half - 1, 1)
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / denom)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2 and t.size(-1) == 1:
            t = t[:, 0]
        t = t.to(self.freqs.dtype)
        angles = t[:, None] * self.freqs[None, :]
        emb = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
        if self.emb_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class MLP(nn.Module):
    def __init__(self, in_shape: tuple[int, int, int] = (1, 28, 28), out_shape: tuple[int, int, int] = (1, 28, 28), hidden_dim: int = 256, t_dim: int = 32):
        super().__init__()
        in_dim = in_shape[0] * in_shape[1] * in_shape[2]
        out_dim = out_shape[0] * out_shape[1] * out_shape[2]
        self.time_mlp = nn.Sequential(
            SinTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.ReLU(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim + t_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.Unflatten(1, out_shape)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(t)
        x_flat = x.view(x.size(0), -1)
        x_in = torch.cat([x_flat, emb], dim=1)
        return self.net(x_in)



class UnetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x, t):
        return self.unet(t, x)

if __name__ == "__main__":
    B, C, H, W = 4, 1, 28, 28
    x = torch.randn(B, C, H, W)
    t = torch.rand(B)

    model = MLP(in_shape=(C, H, W), out_shape=(C, H, W), hidden_dim=256, t_dim=32)
    y = model(x, t)
    print("MLP output shape:", y.shape)

    sin = SinTimeEmbedding(2)
    t_emb = sin(torch.tensor([0.0, 0.05, 0.1, 0.5, 10.0]))
    print("Time embedding shape:", t_emb)
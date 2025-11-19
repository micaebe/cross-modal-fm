import torch
from torch import Tensor
from torchdiffeq import odeint


def flow_pair(
    x0: Tensor, x1: Tensor, t: Tensor, sigma_min: float = 0.0, v_pred: bool = True
) -> tuple[Tensor, Tensor]:
    t = t[:, None, None, None]
    xt = (1.0 - (1.0 - sigma_min) * t) * x0 + t * x1
    if v_pred:
        v_target = x1 - (1.0 - sigma_min) * x0
        return xt, v_target
    else:
        x_target = x1
        return xt, x_target


def integrate_ode(model, x0: Tensor, t0: float, t1: float, steps: int, v_pred: bool = True) -> Tensor:
    device = x0.device

    def f(t, x):
        B = x.shape[0]
        t_b = torch.full((B,), float(t), device=device)
        if v_pred:
            return model(x, t_b)
        else:
            return (model(x, t_b) - x) / (1 - max(0, t - 1e-3))

    ts = torch.linspace(t0, t1, steps, device=device)
    with torch.no_grad():
        xs = odeint(func=f, y0=x0, t=ts, method="euler", atol=1e-5, rtol=1e-5)
    return xs



if __name__ == "__main__":
    B, C, H, W = 4, 1, 28, 28
    x0 = torch.randn(B, C, H, W)
    x1 = torch.randn(B, C, H, W)
    t = torch.rand(B)

    xt, v_target = flow_pair(x0, x1, t)
    print("xt shape:", xt.shape)
    print("v_target shape:", v_target.shape)

    class DummyModel(torch.nn.Module):
        def forward(self, x, t):
            assert x.shape == (B, C, H, W)
            assert t.shape == (B,)
            return x

    model = DummyModel()
    x_end = integrate_ode(model, x0, t0=0.0, t1=1.0, steps=10)
    print("x_end shape:", x_end.shape)
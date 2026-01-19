import torch
from torchdiffeq import odeint

from flowmatch.label_embeddings import (
    RectangleEmbedding,
    OrthoEmbedding,
    GrayScaleEmbedding,
    ClipEmbedding,
)



from flowmatch.dit import DiT_Llama



# RF class from: https://github.com/cloneofsimo/minRF/blob/main/rf.py
# modified to support different source and target distributions, bidirectional training
# and CrossFlow style classifier-free guidance
# sample function is using ODE integration from torchdiffeq
# sin-cos interpolation option is added
class RF:
    def __init__(self,
                 model,
                 ln=False,
                 ln_loc=0.0,
                 ln_scale=1.0,
                 use_sin_cos=False,
                 source="noise",
                 target="image",
                 embedding_type="rectangle",
                 emb_std_scale=0.5,
                 emb_norm_mode="none",
                 img_dim=(32, 32, 1),
                 lambda_b=0.5,
                 bidirectional=False):
        # source and target can be "image", "noise, or "label"
        self.model: DiT_Llama = model
        self.ln = ln
        self.ln_loc = ln_loc
        self.ln_scale = ln_scale
        self.use_sin_cos = use_sin_cos
        self.source_type = source
        self.target_type = target
        self.lambda_b = lambda_b
        self.is_bidirectional = bidirectional
        self.label_embedder = build_embedding_provider(embedding_type,
                                                       H=img_dim[0],
                                                       W=img_dim[1],
                                                       C=img_dim[2],
                                                       num_classes=10,
                                                       std_scale=emb_std_scale,
                                                       mode=emb_norm_mode)

    def get_distribution(self, x, labels, name):
        if name not in ["source", "target"]:
            raise ValueError("name must be either 'source' or 'target'")
        dtype = self.source_type if name == "source" else self.target_type
        if dtype == "image":
            return x
        elif dtype == "noise":
            return torch.randn_like(x)
        elif dtype == "label":
            return self.label_embedder.sample(labels).to(x.device)
        else:
            raise ValueError("dtype must be either 'image', 'noise', or 'label'")

    def forward(self, imgs, labels, cond=None, include_metadata=False):
        b = imgs.size(0)
        if self.ln:
            nt = torch.randn((b,)).to(imgs.device)
            nt = nt * self.ln_scale + self.ln_loc
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(imgs.device)
        texp = t.view([b, *([1] * len(imgs.shape[1:]))])
        #texp = 1.0 - texp
        
        z0 = self.get_distribution(imgs, labels, "source")
        z1 = self.get_distribution(imgs, labels, "target")
        
        bidi_mask = torch.zeros((b,), dtype=torch.bool, device=imgs.device)
        if self.is_bidirectional:
            # False/0 for forward, True/1 for backward
            bidi_mask = torch.rand((b,)).to(imgs.device) > (1.0 - self.lambda_b)
            mask_expanded = bidi_mask.view([b, *([1] * len(imgs.shape[1:]))])
            # swap
            z0_new = torch.where(mask_expanded, z1, z0)
            z1_new = torch.where(mask_expanded, z0, z1)
            z0, z1 = z0_new, z1_new

        if self.use_sin_cos:
            theta = texp * (torch.pi / 2)
            alpha_t = torch.cos(theta)
            beta_t = torch.sin(theta)
            
            zt = alpha_t * z0 + beta_t * z1
            
            d_alpha = - (torch.pi / 2) * beta_t
            d_sigma = (torch.pi / 2) * alpha_t
            target_v = d_alpha * z0 + d_sigma * z1
        else:
            zt = (1 - texp) * z0 + texp * z1
            target_v = z1 - z0

        vtheta = self.model(zt, t, cond, bidi_mask.long() if self.is_bidirectional else None)
        if include_metadata:
            return vtheta, target_v, bidi_mask, t
        return vtheta


    @torch.no_grad()
    def sample(self, z, cond=None, null_cond=None, sample_steps=50, cfg=2.0, direction="forward"):
        b = z.size(0)
        bidi_mask = None
        if self.is_bidirectional:
            label = 0 if direction == "forward" else 1
            bidi_mask = torch.full((b,), label, device=z.device, dtype=torch.long)
            t0, t1 = (0.0, 1.0)
        else:
            t0, t1 = (0.0, 1.0) if direction == "forward" else (1.0, 0.0)
        t_span = torch.linspace(t0, t1, sample_steps + 1, device=z.device)

        def ode_func(t, x):
            t_batch = torch.full((x.shape[0],), t.item(), device=x.device)
            with torch.autocast(dtype=torch.bfloat16):
                vc = self.model(x, t_batch, cond, bidi_mask)
            if null_cond is not None and cfg != 1.0:
                vu = self.model(x, t_batch, null_cond, bidi_mask)
                vc = vu + cfg * (vc - vu)
            return vc
        method = "rk4" if self.use_sin_cos else "euler"
        traj = odeint(ode_func, z, t_span, method=method, atol=1e-5, rtol=1e-5)
        return traj
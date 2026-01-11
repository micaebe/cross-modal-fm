from regex import B
import torch
from torchdiffeq import odeint
from models.dit import DiT_Llama


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
                 source="noise",
                 target="image",
                 label_embedder=None,
                 img_dim=(32, 32, 1),
                 lambda_b=0.5,
                 bidirectional=False,
                 cfg_dropout_prob=0.1,  # only relevant if we are in CrossFlow regime
                 use_conditioning=False
                 ):
        # source and target can be "image", "noise, or "label"
        self.model: DiT_Llama = model
        self.ln = ln
        self.ln_loc = ln_loc
        self.ln_scale = ln_scale
        self.source_type = source
        self.target_type = target
        self.lambda_b = lambda_b
        self.is_bidirectional = bidirectional
        self.label_embedder = label_embedder
        self.cfg_dropout_prob = cfg_dropout_prob
        self.use_conditioning = use_conditioning

    def get_endpoints(self, imgs, labels, name):
        # helper to resolve the correct endpoints
        if name not in ["source", "target"]:
            raise ValueError("name must be either 'source' or 'target'")
        dtype = self.source_type if name == "source" else self.target_type
        if dtype == "image":
            return imgs
        elif dtype == "noise":
            return torch.randn_like(imgs)
        elif dtype == "label":
            return self.label_embedder(labels).to(imgs.device)
        else:
            raise ValueError("dtype must be either 'image', 'noise', or 'label'")
    
    def _get_conditioning(self, labels):
        # helper for CrossFlow style classifier-free-guidance
        cond = None # adaln conditioning
        if self.use_conditioning:
            cond = labels
        else:
            # CrossFlow style indicators for beeing able to sample unconditionally and conditionally (to enable cfg)
            to_drop = (torch.rand(labels.size(0), device=labels.device) < self.cfg_dropout_prob)
            labels = labels.clone()
            labels[to_drop] = torch.randint(0, self.label_embedder.num_classes, (to_drop.sum(),), device=labels.device)
            cond = to_drop.long() # -> cfg indicator
        # cond is the conditioning inside model (eg adaln) -> = CFG-indicator if we are in CrossFlow regime; = labels if normal FM regime
        # labels is used for retrieving the label embeddings only, includes the class dropout for CrossFlow style cfg
        return cond, labels
    
    def _get_bidir_inputs(self, z0, z1):
        # helper to get swapped endpoints and bidirectional mask
        # False/0 for forward, True/1 for backward
        bidi_mask = torch.rand((z0.shape[0],), device=z0.device) > (1.0 - self.lambda_b)
        mask_expanded = bidi_mask.view([z0.shape[0], *([1] * len(z0.shape[1:]))])
        # swap
        z0_new = torch.where(mask_expanded, z1, z0)
        z1_new = torch.where(mask_expanded, z0, z1)
        return z0_new, z1_new, bidi_mask


    def forward(self, imgs, labels, include_metadata=False, t=None, bidi_mask=None):
        b = imgs.size(0)
        if t is None:
            if self.ln:
                nt = torch.randn((b,), device=imgs.device)
                nt = nt * self.ln_scale + self.ln_loc
                t = torch.sigmoid(nt)
            else:
                t = torch.rand((b,), device=imgs.device)
        texp = t.view([b, *([1] * len(imgs.shape[1:]))])

        cond, labels = self._get_conditioning(labels)
        z0 = self.get_endpoints(imgs, labels, "source")
        z1 = self.get_endpoints(imgs, labels, "target")

        if self.is_bidirectional:
            z0, z1, bidi_mask = self._get_bidir_inputs(z0, z1)

        # linear path
        zt = (1 - texp) * z0 + texp * z1
        target_v = z1 - z0

        vtheta = self.model(zt, t, cond, bidi_mask.long() if self.is_bidirectional else None)
        if include_metadata:
            return vtheta, target_v, bidi_mask, t
        return vtheta
    

    @torch.no_grad()
    def sample(self, z, cond=None, null_cond=None, sample_steps=50, cfg=2.0, direction="forward", invert_time=False):
        """
        Args:
            z: image, noise or label embedding
            cond: discrete conditioning signal, class labels in case of "normal" FM, cfg indicators in case of CrossFlow FM
            null_cond: discrete null conditioning signal
            sample_steps: number of sampling steps
            cfg: classifier-free guidance scale
            direction: "forward" or "backward"
            invert_time: Whether to invert the time variable. Only applied in case the model is bidirectional
        """
        b = z.size(0)
        bidi_mask = None
        if self.is_bidirectional:
            dir_label = 0 if direction == "forward" else 1
            bidi_mask = torch.full((b,), dir_label, device=z.device, dtype=torch.long)
            t0, t1 = (0.0, 1.0)
            if invert_time:
                t0, t1 = (1.0, 0.0)
        else:
            t0, t1 = (0.0, 1.0) if direction == "forward" else (1.0, 0.0)
        t_span = torch.linspace(t0, t1, sample_steps + 1, device=z.device)

        def ode_func(t, x):
            t_batch = torch.full((x.shape[0],), t.item(), device=x.device)
            if null_cond is not None and cfg != 1.0:
                x_in = torch.cat([x, x])
                t_in = torch.cat([t_batch, t_batch])
                cond_in = torch.cat([cond, null_cond])
                bidi_in = torch.cat([bidi_mask, bidi_mask]) if bidi_mask is not None else None
                
                v_out = self.model(x_in, t_in, cond_in, bidi_in)
                vc, vu = v_out.chunk(2)
                vc = vu + cfg * (vc - vu)
            else:
                vc = self.model(x, t_batch, cond, bidi_mask)
            return vc
        method = "euler" # "rk4"
        device_type = z.device.type
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=True if device_type == 'cuda' else False):
            traj = odeint(ode_func, z, t_span, method=method, atol=1e-5, rtol=1e-5)
        return traj
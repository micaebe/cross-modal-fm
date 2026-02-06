import torch
from torchdiffeq import odeint


# RF class inspired from: https://github.com/cloneofsimo/minRF/blob/main/rf.py
# modified to support different source and target distributions & bidirectional training
# and CrossFlow style classifier-free guidance
# sample function is using ODE integration from torchdiffeq
class RF:
    def __init__(self,
                 model,
                 ln=False,
                 ln_loc=0.0,
                 ln_scale=1.0,
                 source="noise",
                 target="image",
                 label_embedder=None,
                 lambda_b=0.5,
                 bidirectional=False,
                 cls_dropout_prob=0.1,  # only relevant if we are in CrossFlow regime
                 use_conditioning=False,
                 ode_method="euler",
                 stochastic_interpolant_scale=0.0 # 0.0 = RF/FM, > 0.0 = Stochastic Interpolant
                 ):
        """
        Initializes the RF class.

        Args:
            model: The model to use (DiT/UNet)
            ln: Whether to use logit-normal timestep sampling.
            ln_loc: The location parameter for logit-normal sampling (only used if ln is True).
            ln_scale: The scale parameter for logit-normal sampling (only used if ln is True).
            source: The source distribution type ("image", "noise", or "label"). Label is the class representation.
            target: The target distribution type ("image", "noise", or "label"). Label is the class representation.
            label_embedder: The label embedder to use. (Class representations)
            lambda_b: The bidirectional mixing ratio (lambda_m in the report)
            bidirectional: Whether to use bidirectional training.
            cls_dropout_prob: The class dropout probability (CrossFlow style indicators) 
            use_conditioning: Whether to use conditioning. This should be usually False in cross-modal regime.
            ode_method: The ODE integration method to use, e.g. euler, rk4
            stochastic_interpolant_scale: Optionally one can use a stochastic bridge by setting this to a value > 0 (wasnt used in the report).
        """
        # source and target can be "image", "noise, or "label"
        self.model = model
        self.ln = ln
        self.ln_loc = ln_loc
        self.ln_scale = ln_scale
        self.source_type = source
        self.target_type = target
        self.lambda_b = lambda_b
        self.bidirectional = bidirectional
        self.label_embedder = label_embedder
        self.cls_dropout_prob = cls_dropout_prob
        self.use_conditioning = use_conditioning
        self.ode_method = ode_method
        self.si_scale = stochastic_interpolant_scale

    @property
    def is_forward_model(self):
        """
        Determine if the model is in C2I or I2C training mode.
        """
        return self.target_type == "image"

    def resolve_endpoints(self, imgs, labels, endpoint, embedding_noise=None):
        """
        Helper to get the source or target distribution.
        """
        if endpoint not in ["source", "target"]:
            raise ValueError("endpoint type must be either 'source' or 'target'")
        endpoint_type = self.source_type if endpoint == "source" else self.target_type
        if endpoint_type == "image":
            return imgs
        elif endpoint_type == "noise":
            return torch.randn_like(imgs)
        elif endpoint_type == "label":
            sample_internal = embedding_noise is None
            emb = self.label_embedder(labels, sample_internal)
            if embedding_noise is not None:
                emb += embedding_noise * self.label_embedder.std_scale
            return emb
        else:
            raise ValueError("unknown endpoint type")
    
    def _prepare_conditioning(self, labels):
        """
        Prepares cross-flow style indicator function & applies class dropout

        Returns (model_conditioning, label_indices_for_embedding)
        """
        if self.use_conditioning:
            # use_conditioning = use model conditioning only 
            return labels, labels
        # crossflow style indicators
        to_drop = (torch.rand(labels.size(0), device=labels.device) < self.cls_dropout_prob)
        masked_labels = labels.clone()
        num_classes = self.label_embedder.num_classes
        masked_labels[to_drop] = torch.randint(0, num_classes, (to_drop.sum(),), device=labels.device)
        return to_drop.long(), masked_labels

    def _get_bidir_inputs(self, z0, z1):
        """
        Helper to get swapped endpoints and bidirectional mask.
        False/0 for forward, True/1 for backward.
        """
        # helper to get swapped endpoints and bidirectional mask
        # False/0 for forward, True/1 for backward
        b = z0.shape[0]
        bidi_mask = torch.rand((b,), device=z0.device) > (1.0 - self.lambda_b)
        mask_expanded = bidi_mask.view([b, *([1] * len(z0.shape[1:]))])
        # swap
        z0_new = torch.where(mask_expanded, z1, z0)
        z1_new = torch.where(mask_expanded, z0, z1)
        return z0_new, z1_new, bidi_mask.long()


    def forward(self, imgs, labels, include_metadata=False, t=None, bidi_mask=None):
        """
        Forward pass for the model.

        Args:
            imgs: The input images.
            labels: The corresponding class labels.
            include_metadata: Whether to return additional metadata (bidi_mask and t).
            t: Optional timesteps. If None, they will be sampled (logit-normal if ln is True, else uniform).
        """
        b = imgs.size(0)
        if t is None:
            if self.ln:
                nt = torch.randn((b,), device=imgs.device)
                nt = nt * self.ln_scale + self.ln_loc
                t = torch.sigmoid(nt)
            else:
                t = torch.rand((b,), device=imgs.device)
            if not self.is_forward_model:
                #  in case we are training image -> label,
                # we flip the time. This ensures that, with same seed,
                # both directions see the same samples during training.
                t = 1.0 - t
        texp = t.view([b, *([1] * len(imgs.shape[1:]))])

        cond, labels = self._prepare_conditioning(labels)
        z0 = self.resolve_endpoints(imgs, labels, "source")
        z1 = self.resolve_endpoints(imgs, labels, "target")

        bidi_mask = None
        if self.bidirectional:
            # if the model is bidirectional, we swap endpoints randomly (weighted by lambda_b)
            # bidi_mask indicates which samples are swapped
            z0, z1, bidi_mask = self._get_bidir_inputs(z0, z1)

        # linear path
        zt = (1 - texp) * z0 + texp * z1
        # d/dt zt
        target_v = z1 - z0


        if self.si_scale > 0.0:
            # stochastic interpolant (can be optionally added)
            noise = torch.randn_like(zt) * self.si_scale
            # zt = zt + stochastic_bridge
            # stochastic_bridge = t * (1 - t) * noise
            zt = zt + texp * (1.0 - texp) * noise
            # d/dt stochastic bridge: (1 - 2 * t) * noise
            target_v = target_v + (1 - 2 * texp) * noise

        vtheta = self.model(zt, t, cond, bidi_mask)
        if include_metadata:
            return vtheta, target_v, bidi_mask, t
        return vtheta
    

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0, direction="forward", invert_time=False):
        """
        Args:
            z: image, noise or label embedding
            cond: discrete conditioning signal, class labels in case of "normal" FM, cfg indicators in case of CrossFlow FM
            null_cond: discrete null conditioning signal (only used in case cfg > 1.0)
            sample_steps: number of sampling steps
            cfg: classifier-free guidance scale
            direction: "forward" or "backward"
            invert_time: Whether to invert the time variable. Only applied in case the model is bidirectional
        """
        b = z.size(0)

        t0, t1 = (0.0, 1.0)
        if (direction == "backward" and not self.bidirectional) or (self.bidirectional and invert_time):
            t0, t1 = (1.0, 0.0)

        bidi_mask = None
        if self.bidirectional:
            # "standard" mode in bidirectional case is always from t0=0 to t1=1
            # and the bidirectional mask controls the direction
            dir_label = 0 if direction == "forward" else 1
            bidi_mask = torch.full((b,), dir_label, device=z.device, dtype=torch.long)
        t_span = torch.linspace(t0, t1, sample_steps + 1, device=z.device)

        def ode_func(t, x):
            t_batch = torch.full((x.shape[0],), t.item(), device=x.device)
            if null_cond is not None and cfg != 1.0:
                # ise classifier free guidance
                x_in = torch.cat([x, x])
                t_in = torch.cat([t_batch, t_batch])
                cond_in = torch.cat([cond, null_cond])
                bidi_in = torch.cat([bidi_mask, bidi_mask]) if bidi_mask is not None else None
                
                v_out = self.model(x_in, t_in, cond_in, bidi_in)
                vc, vu = v_out.chunk(2)
                vc = vu + cfg * (vc - vu)
            else:
                # no classifier free guidance
                vc = self.model(x, t_batch, cond, bidi_mask)
            return vc
        device_type = z.device.type
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=True if device_type == 'cuda' else False):
            traj = odeint(ode_func, z, t_span, method=self.ode_method, atol=1e-5, rtol=1e-5)
        return traj
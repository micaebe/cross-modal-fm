import torch

class EMA:
    """
    Exponential moving average with optional warmup period.
    """
    def __init__(self, model, decay=0.999, warmup_steps=0):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step = 0
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.clone().detach()

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                if self.step <= self.warmup_steps:
                    self.shadow[name].copy_(param.detach())
                else:
                    self.shadow[name].lerp_(param.detach(), 1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.clone().detach()
                param.data.copy_(self.shadow[name].data)

    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name].data)
        self.backup = {}

    def state_dict(self):
        return {
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "step": self.step,
            "shadow": self.shadow
        }

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.step = state_dict["step"]
        
        for name, param in state_dict["shadow"].items():
            if name in self.shadow:
                self.shadow[name].copy_(param)
            else:
                self.shadow[name] = param
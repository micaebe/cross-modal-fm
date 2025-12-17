# very bad/hacky script to run FID evaluations on saved models

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
np.int = int
from torch_fidelity import calculate_metrics
from torch_fidelity.generative_model_base import GenerativeModelBase

from dit import DiT_Llama
from rf import RF

def parse_config_from_dirname(dirname):
    parts = dirname.split('_')
    args = argparse.Namespace()
    # 0: cifar, 1: source, 2: to, 3: target, 4: cond, 5: bidir
    args.source = parts[1]
    args.target = parts[3]
    args.use_conditioning = parts[4] == "True"
    args.bidirectional = parts[5] == "True"

    args.embedding_type = None
    args.emb_std_scale = 0.2
    args.emb_norm_mode = "none"

    if args.source != "noise" and len(parts) >= 9:
        args.embedding_type = parts[6]
        
        std_str = parts[7]
        if std_str.startswith("std"):
            val_str = std_str.replace("std", "")
            try:
                if "." in val_str:
                    args.emb_std_scale = float(val_str)
                else:
                    args.emb_std_scale = float(val_str) / 10.0
            except ValueError:
                raise ValueError("Error parsing emb_std_scale from directory name.")
        
        args.emb_norm_mode = parts[8]

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.use_ln = False
    args.ln_loc = 0.0
    args.ln_scale = 1.0
    args.use_sin_cos = False
    
    return args

def load_compiled_checkpoint(model, path):
    state_dict = torch.load(path, map_location='cpu')
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    return model

def load_model_architecture(args):
    image_dims = (32, 32, 3)
    H, W, C = image_dims
    
    num_classes = 11
    if not args.use_conditioning:
        num_classes = 2
    if args.bidirectional:
        num_classes = 2

    model_kwargs = dict(
        in_channels=C,
        input_size=H,
        patch_size=2,
        dim=64 if C == 1 else 256,
        n_layers=6 if C == 1 else 10,
        n_heads=8,
        class_dropout_prob=0.0,
        num_classes=num_classes,
        bidirectional=args.bidirectional
    )
    return DiT_Llama(**model_kwargs).to(args.device)

class FMFIDWrapper(GenerativeModelBase):
    def __init__(self, rf, num_classes=10, image_size=32, device="cuda", 
                 steps=100, cfg=1.0, direction="backward"):
        super().__init__()
        self._num_classes = num_classes
        self.image_size = image_size
        self.device = device
        self.rf = rf
        
        self.steps = steps
        self.cfg = cfg
        self.direction = direction

    @property
    def z_size(self): return 1
    @property
    def z_type(self): return "normal"
    @property
    def num_classes(self): return self._num_classes

    @torch.no_grad()
    def forward(self, x, y=None):
        if y is None:
            y = torch.randint(0, self.num_classes, (x.size(0),), device="cpu").long()

        if self.rf.source_type != "noise":
            z = self.rf.label_embedder.sample(y.to("cpu"), sample=True).to(self.device)
            cond = torch.zeros_like(y).long().to(self.device)
            null_cond = torch.ones_like(y).long().to(self.device)
        else:
            z = torch.randn(x.size(0), 3, 32, 32).to(self.device) 
            cond = y.to(self.device)
            null_cond = torch.full_like(y, self.num_classes).to(self.device)


        out = self.rf.sample(
            z, cond, null_cond, 
            sample_steps=self.steps, 
            cfg=self.cfg, 
            direction=self.direction
        )
        
        imgs = out[-1, :]
        
        imgs = imgs.float()
        if imgs.min() < 0:
            imgs = (imgs + 1.0) / 2.0
            
        return (imgs.clamp(0.0, 1.0) * 255.0).to(torch.uint8)


# adjust parameters as needed
def run_evaluation():
    ROOT_DIR = "training_outputs"
    TARGET_EPOCHS = [50, 100, 200, 300, 350, 400] #[100, 200, 300, 400]
    STEP_LIST = [100] #[10, 40]
    CFG_LIST = [2.0] #[1.0, 2.0]
    FID_BATCH_SIZE = 1024
    NUM_SAMPLES = 10000
    
    subdirs = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
    
    print(f"Found {len(subdirs)} runs to evaluate.")

    for dirname in subdirs:
        model_dir_path = os.path.join(ROOT_DIR, dirname)
        print(f"\nProcessing: {dirname}")
        
        try:
            args = parse_config_from_dirname(dirname)
            print(f"Args: source={args.source}, target={args.target}, conditioning={args.use_conditioning}, bidirectional={args.bidirectional}, embedding_type={args.embedding_type}, emb_std_scale={args.emb_std_scale}, emb_norm_mode={args.emb_norm_mode}")
        except Exception as e:
            print(f"Skipping {dirname} due to parsing error: {e}")
            continue

        direction = "backward" if args.source == "image" else "forward"
        
        results = []
        
        for epoch in TARGET_EPOCHS:
            ckpt_name = f"model_ema{epoch}.pt"
            ckpt_path = os.path.join(model_dir_path, ckpt_name)
            
            if not os.path.exists(ckpt_path):
                print(f"Checkpoint {ckpt_name} not found. Skipping.")
                continue
                
            print(f"Loading Epoch {epoch}...")
            
            try:
                model = load_model_architecture(args)
                model = load_compiled_checkpoint(model, ckpt_path)
                model.eval()
            except Exception as e:
                print(f"Error loading model: {e}")
                continue

            rf_ema = RF(
                model=model,
                source=args.source,
                target=args.target,
                ln=args.use_ln,
                ln_loc=args.ln_loc,
                ln_scale=args.ln_scale,
                embedding_type=args.embedding_type if args.embedding_type != None else "rectangle",
                emb_std_scale=args.emb_std_scale,
                emb_norm_mode=args.emb_norm_mode,
                bidirectional=args.bidirectional,
                use_sin_cos=args.use_sin_cos,
                img_dim=(32, 32, 3),
            )

            for steps in STEP_LIST:
                for cfg in CFG_LIST:
                    print(f"Evaluating: Epoch={epoch}, Steps={steps}, CFG={cfg}")
                    fid_wrapper = FMFIDWrapper(
                        rf=rf_ema,
                        image_size=32,
                        device=args.device,
                        steps=steps,
                        cfg=cfg,
                        direction=direction
                    )
                    try:
                        metrics = calculate_metrics(
                            input1=fid_wrapper,
                            input2="cifar10-val",
                            fid=True,
                            cuda=True,
                            batch_size=FID_BATCH_SIZE,
                            input1_model_num_samples=NUM_SAMPLES,
                            cache=False
                        )
                        
                        fid_score = metrics["frechet_inception_distance"]                        
                        results.append({
                            "epoch": epoch,
                            "steps": steps,
                            "cfg": cfg,
                            "fid": fid_score
                        })
                    except Exception as e:
                        print(f"Error calculating FID: {e}")

        if results:
            df = pd.DataFrame(results)
            csv_path = os.path.join(model_dir_path, "fid_results.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved results to {csv_path}")

if __name__ == "__main__":
    run_evaluation()
# modified to support:
# - unconditional training
# - swapping source and target (= training forward and backward)
# - using a label embedding as source/target
# - x-prediction with v-loss. This resembles (a)(3) from https://arxiv.org/pdf/2511.13720
# - bidirectional training using conditioning on direction with configurable weighting between forward and backward training.

import argparse
import os

import torch
from flowmatch.label_embeddings import ScalarLevelsEmbedding


class RF:
    def __init__(self, model, ln=True, source="noise", target="image", use_conditioning=False, prediction="v", lambda_b=0.5, bidirectional=False):
        """
        Args:
            source: the source distribution, either "noise", "image" or "label_embedding"
            target: the target distribution, either "noise", "image" or "label_embedding"
            use_conditioning: if we want to use class conditioning or not (can only be used if bidirectional=False)
            prediction: either "v" or "x", specifies what we predict ("v" = velocity, "x" = data). We always use v-loss.
            lambda_b: weighting between forward and backward if bidirectional, in range (0, 1). 0 == only forward, 1 == only backward.
            bidirectional: train in both directions if True. Weighting: `1-lambda` forward (label 0) and `lambda` backward (label 1).
        """
        self.model = model
        self.ln = ln
        self.source_type = source
        self.target_type = target
        self.use_conditioning = use_conditioning
        self.prediction = prediction
        self.lambda_b = lambda_b
        self.bidirectional = bidirectional
        self.label_embedder = ScalarLevelsEmbedding(H=32, W=32, C=1, num_classes=10, std_scale=0.3)

        if self.bidirectional and self.use_conditioning:
            raise ValueError("Cannot use class conditioning with bidirectional training (class input is used for direction flag).")

    def get_distribution(self, dtype, x, cond):
        if dtype == "image":
            return x
        elif dtype == "noise":
            return torch.randn_like(x)
        elif dtype == "label_embed":
            return self.label_embedder.sample(cond).to(cond.device)

    def forward(self, x, cond):
        b = x.size(0)
        if self.ln:
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        
        z0 = self.get_distribution(self.source_type, x, cond)
        z1 = self.get_distribution(self.target_type, x, cond)
        
        bwd_mask = torch.zeros((b,), dtype=torch.bool, device=x.device)
        if self.bidirectional:
            # False/0 for forward, True/1 for backward
            bwd_mask = torch.rand((b,)).to(x.device) > (1.0 - self.lambda_b)
            cond = bwd_mask.long()
            mask_expanded = bwd_mask.view([b, *([1] * len(x.shape[1:]))])
            # swap
            z0_new = torch.where(mask_expanded, z1, z0)
            z1_new = torch.where(mask_expanded, z0, z1)
            z0, z1 = z0_new, z1_new
        elif not self.use_conditioning:
            cond *= 0
        
        zt = (1 - texp) * z0 + texp * z1
        model_output = self.model(zt, t, cond)
        
        if self.prediction == "v":
            vtheta = model_output
        elif self.prediction == "x":
            # v_theta = (x_theta - z_t) / (1 - t)
            vtheta = (model_output - zt) / (1.0 - texp + 1e-4)

        batchwise_mse = ((z1 - z0 - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        dlist = bwd_mask.float().detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, dir, tloss) for tv, dir, tloss in zip(t, dlist, tlist)]
        return batchwise_mse.mean(), ttloss

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0, direction="forward"):
        """
        direction: 'forward' (source -> target, t=0->1) or 'backward' (target -> source, t=1->0)
        """
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]

        if self.bidirectional:
            label = 0 if direction == "forward" else 1
            cond = torch.full((b,), label, device=z.device, dtype=torch.long)
        
        for i in range(sample_steps):
            if direction == "forward" or self.bidirectional:
                # in case of bidirectional, we always sample from t=0 to t=1 and use the directional conditioning to handle direction
                t = i / sample_steps
                step_sign = 1.0
            else:
                t = 1.0 - (i / sample_steps)
                step_sign = -1.0
                
            t_tensor = torch.tensor([t] * b).to(z.device)

            cond_run = cond.clone()
            if not self.use_conditioning and not self.bidirectional:
                cond_run *= 0
            
            out_c = self.model(z, t_tensor, cond_run)
            
            if self.prediction == "v":
                vc = out_c
            elif self.prediction == "x":
                # v = (x - z) / (1 - t)
                denom = 1.0 - t
                if abs(denom) < 1e-4: denom = 1e-4
                vc = (out_c - z) / denom

            if null_cond is not None and not self.bidirectional:
                out_u = self.model(z, t_tensor, null_cond)
                if self.prediction == "v":
                    vu = out_u
                elif self.prediction == "x":
                    # v = (x - z) / (1 - t)
                    denom = 1.0 - t
                    if abs(denom) < 1e-4: denom = 1e-4
                    vu = (out_u - z) / denom
                vc = vu + cfg * (vc - vu)
            
            z = z + step_sign * dt * vc
            images.append(z)
            
        return images


if __name__ == "__main__":
    # train class conditional RF on mnist.
    import numpy as np
    import torch.optim as optim
    from PIL import Image
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.utils import make_grid
    from tqdm import tqdm

    import wandb
    from dit import DiT_Llama

    parser = argparse.ArgumentParser(description="RF Config")
    parser.add_argument("--cifar", action="store_true")
    parser.add_argument("--source", type=str, default="noise", choices=["noise", "image", "label_embed"])
    parser.add_argument("--target", type=str, default="image", choices=["noise", "image", "label_embed"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v", "x"], help="Prediction target: v (velocity) or x (data)")
    parser.add_argument("--use_conditioning", action="store_true", help="Enable class conditioning")
    parser.add_argument("--bidirectional", action="store_true", help="Enable bidirectional training (overrides conditioning)")
    parser.add_argument("--lambda_b", type=float, default=0.5, help="The weighting parameter used for bidirectional training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda, cpu, mps)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Prediction mode: {args.prediction}-pred with v-loss")

    if args.bidirectional:
        print(f"Bidirectional mode enabled. Num classes set to 2 (0=Fwd, 1=Bwd). Weighting: {((1-args.lambda_b)*100):.2f}% Fwd, {(args.lambda_b*100):.2f}% Bwd")
        num_classes = 2
        args.use_conditioning = False
    else:
        num_classes = 10

    CIFAR = args.cifar

    if CIFAR:
        dataset_name = "cifar"
        fdatasets = datasets.CIFAR10
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        channels = 3
        model = DiT_Llama(
            channels, 32, dim=256, n_layers=10, n_heads=8, num_classes=10
        ).to(device)

    else:
        dataset_name = "mnist"
        fdatasets = datasets.MNIST
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Pad(2),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        channels = 1
        model = DiT_Llama(
            channels, 32, dim=64, n_layers=6, n_heads=4, num_classes=10
        ).to(device)

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6}M")

    rf = RF(model, source=args.source, target=args.target,
            use_conditioning=args.use_conditioning,
            prediction=args.prediction,
            lambda_b=args.lambda_b,
            bidirectional=args.bidirectional)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = torch.nn.MSELoss()

    mnist = fdatasets(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(mnist, batch_size=256, shuffle=True, drop_last=True)

    mnist_val = fdatasets(root="./data", train=False, download=True, transform=transform)
    dataloader_val = DataLoader(mnist_val, batch_size=16, shuffle=False, drop_last=True)

    # used for generating gifs/images in case we start from the image distribution
    batch = next(iter(dataloader_val))
    val_x = batch[0][:16].to(device)
    val_c = batch[1][:16].to(device)

    run_name = f"{args.source}_to_{args.target}_{args.prediction}_cond_{args.use_conditioning}_bidi_{args.bidirectional}"
    wandb.init(project=f"rf_{dataset_name}", name=run_name, config=args)

    for epoch in range(100):
        if args.bidirectional:
            lossbin = {"fwd": {i: 0 for i in range(10)}, "bwd": {i: 0 for i in range(10)}}
            losscnt = {"fwd": {i: 1e-6 for i in range(10)}, "bwd": {i: 1e-6 for i in range(10)}}
        else:
            lossbin = {i: 0 for i in range(10)}
            losscnt = {i: 1e-6 for i in range(10)}
        for i, (x, c) in tqdm(enumerate(dataloader)):
            x, c = x.to(device), c.to(device)
            optimizer.zero_grad()
            loss, blsct = rf.forward(x, c)
            loss.backward()
            
            if i % 10 == 0:
                grad_stats = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_stats[f"grads/{name}_norm"] = param.grad.norm().item()
                        grad_stats[f"grads/{name}_mean"] = param.grad.mean().item()
                    grad_stats[f"weights/{name}_norm"] = param.data.norm().item()
                wandb.log(grad_stats, commit=False)

            optimizer.step()

            wandb.log({"loss": loss.item()})

            # count based on t and if bidirectional also on d
            for t, d, l in blsct:
                bin_idx = int(t * 10)
                if args.bidirectional:
                    key = "bwd" if d > 0.5 else "fwd"
                    lossbin[key][bin_idx] += l
                    losscnt[key][bin_idx] += 1
                else:
                    lossbin[bin_idx] += l
                    losscnt[bin_idx] += 1

        if args.bidirectional:
            log_dict = {}
            for i in range(10):
                val = lossbin["fwd"][i] / losscnt["fwd"][i]
                print(f"Epoch: {epoch}, FWD {i} range loss: {val}")
                log_dict[f"lossbin_fwd_{i}"] = val

                val = lossbin["bwd"][i] / losscnt["bwd"][i]
                print(f"Epoch: {epoch}, BWD {i} range loss: {val}")
                log_dict[f"lossbin_bwd_{i}"] = val
            wandb.log(log_dict)
        else:
            for i in range(10):
                print(f"Epoch: {epoch}, {i} range loss: {lossbin[i] / losscnt[i]}")
            wandb.log({f"lossbin_{i}": lossbin[i] / losscnt[i] for i in range(10)})


        if epoch % 5 == 0:
            rf.model.eval()
            with torch.no_grad():
                uncond = None # we do not use cfg by default
                save_path = "./contents"
                if not os.path.exists(save_path): os.makedirs(save_path, exist_ok=True)

                # forward
                init_fwd = rf.get_distribution(args.source, val_x, val_c)
                images_fwd = rf.sample(init_fwd, val_c, uncond, direction="forward")
                
                gif_fwd = []
                for image in images_fwd:
                    image = image * 0.5 + 0.5
                    image = image.clamp(0, 1)
                    x_as_image = make_grid(image.float(), nrow=4)
                    img = x_as_image.permute(1, 2, 0).cpu().numpy()
                    img = (img * 255).astype(np.uint8)
                    gif_fwd.append(Image.fromarray(img))
                
                gif_fwd[0].save(
                    f"{save_path}/{run_name}_fwd_{epoch}.gif",
                    save_all=True, append_images=gif_fwd[1:], duration=100, loop=0,
                )
                gif_fwd[-1].save(f"{save_path}/{run_name}_fwd_{epoch}_last.png")

                # backward
                init_bwd = rf.get_distribution(args.target, val_x, val_c)
                images_bwd = rf.sample(init_bwd, val_c, uncond, direction="backward")
                
                gif_bwd = []
                for image in images_bwd:
                    image = image * 0.5 + 0.5
                    image = image.clamp(0, 1)
                    x_as_image = make_grid(image.float(), nrow=4)
                    img = x_as_image.permute(1, 2, 0).cpu().numpy()
                    img = (img * 255).astype(np.uint8)
                    gif_bwd.append(Image.fromarray(img))
                
                gif_bwd[0].save(
                    f"{save_path}/{run_name}_bwd_{epoch}.gif",
                    save_all=True, append_images=gif_bwd[1:], duration=100, loop=0,
                )
                gif_bwd[-1].save(f"{save_path}/{run_name}_bwd_{epoch}_last.png")

        rf.model.train()
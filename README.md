# Cross-Modal Flow Matching

Cross-modal flow matching between label/class representations and images. Supports two 'unidirectional' (class-to-image, image-to-class / C2I, I2C) and a bidirectional training mode on MNIST, CIFAR-10, and ImageNet-100.



## Setup

Optionally first create and activate a new virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```



## Usage

Train a model with defaults (CIFAR-10, class->image, DiT):
```bash
python -m xfm.main
```


Run a predefined experiment sweep:
```bash
python -m xfm.main +experiment=cifar
python -m xfm.main +experiment=mnist
python -m xfm.main +experiment=imagenet
```

Note: For ImageNet, the dataset needs to be downloaded and processed beforehand by running `download_imagenet.sh`.
Also note: we used 'label' and 'class' and also 'latent' and 'embedding' interchangeably for the class representations.


### Key Configuration Options

Some of the more important options:
- `dataset`: `mnist`, `cifar` (default), or `imagenet100`
- `model`: `DiT` (default) or `UNet`
- `mode`: `l2i` (label/class->image, default), `i2l` (image->class/label), or `bidi` (bidirectional)
- `embeddings.name`: label embedding type, `rectangle` (default), `grayscale`, `smoothrand`, `clip`
- `embeddings.std_scale`: noise scale for label embeddings (default `1.0`)
- `rf.ln`: use logit-normal timestep sampling (`rf.ln_loc`, `rf.ln_scale` control its parameters)
- `rf.lambda_b`: ratio of mixing directions in bidirectional training (default `0.5`; this is $\lambda_m$)
- `use_conditioning`: if `true`, uses standard class-conditional CFG instead of CrossFlow style indicators
- `mixed_precision` / `compile_model`: bfloat16 training and `torch.compile` (both `true` by default)
- `total_steps`: total training iterations (default `300000`)
- `eval_cfg_scale`: CFG scale used during evaluation (default `1.0`)

See `xfm/conf/` for the full set of configs and defaults.



## Code Structure

Everything lives under the `xfm/` package. The main entry points are `main.py` (training) and `evaluate_checkpoints.py` (post evaluation of saved checkpoints).

- `config.py` configures the model, optimizer, EMA, dataloaders, etc.
- `rf.py` contains the core Rectified Flow/Flow Matching logic.
- `train.py` contains the training loop.
- `utils.py` has helpers for seeding and checkpoint saving/loading.
- `conf/` contains the hydra configs.
- `dataset/` handles dataset construction.
- `embeddings/` defines different class/label embedding strategies (class latents).
- `evaluation/` contains the evaluation loop (FID, Precision, Recall), plus a small ResNet classifier that can optionally be used to measure generation class correspondence (but wasnt used in the final experiments).
- `models/` contains two supported architectures: a DiT (adapted from Fast-DiT) and a UNet (adapted from guided-diffusion), along with a EMA helper.




## Acknowledgements

This project builds on and is inspired by the following works:

- **[minRF](https://github.com/cloneofsimo/minRF)**  The `RF` class in `xfm/rf.py`, is heavily inspired by minRF. Core functionality such as the rectified flow forward pass and logit-normal timestep sampling is adapted from minRF, with modifications for:
  - Cross-modal source/target distributions
  - Bidirectional training
  - CrossFlow-style classifier-free guidance
  - ODE integration via `torchdiffeq` to be able to try different integration methods

- **[Fast-DiT](https://github.com/chuanyangjin/fast-DiT)** The DiT model architecture in `xfm/models/fast_dit.py` is adapted from Fast-DiT, with added support for bidirectional conditioning.

- **[guided-diffusion](https://github.com/openai/guided-diffusion)** The UNet model in `xfm/models/unet.py` is adapted from guided-diffusion, with added support for bidirectional conditioning.


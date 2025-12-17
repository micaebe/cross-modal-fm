# Flow Matching for Image Generation

Exploring flow matching with different label embedding strategies for conditional image generation on MNIST and CIFAR-10.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Train a model:
```bash
python flowmatch/main.py --source label --target image --label_embedding rectangle --dataset mnist --epochs 50 --device cuda
```

Key arguments:
- `--source` / `--target`: Flow direction (`label`, `image`, `noise`)
- `--label_embedding`: Embedding type (`grayscale`, `rectangle`, `ortho`, `clip`)
- `--embedding_std_scale`: Noise scale for label embeddings
- `--bidirectional`: Train bidirectional flow
- `--use_conditioning`: Use class conditioning in DiT
- `--use_ln`, `--ln_loc`, `--ln_scale`: Logit normal timestep sampling during training
- `--use_bf16`: Enable bfloat16 training
- `--wandb`: Enable W&B logging

See `flowmatch/run.sh` for example experiments.


## Acknowledgements

This repository uses code from [minRF](https://github.com/cloneofsimo/minRF) by Simo Ryu.
Specifically, the `DiT_Llama` model architecture in `flowmatch/dit.py` and the core Rectified Flow implementation in `flowmatch/rf.py` are adapted from `minRF`, main modifications include:
- Different source/target distributions (Label, Image)
- Bidirectional training
- CrossFlow-style classifier-free guidance


## Code Structure

- `flowmatch/main.py`: Entry point for training. Handles argument parsing, data loading, and model initialization.
- `flowmatch/train.py`: Contains the training loop and evaluation logic.
- `flowmatch/rf.py`: Core Rectified Flow implementation. Defines the forward pass (loss calculation) and sampling (ODE integration).
- `flowmatch/dit.py`: The DiT-Llama model architecture.
- `flowmatch/label_embeddings.py`: Defines various label embedding strategies (Grayscale, Rectangle, Ortho, CLIP) that map class labels to image-space tensors.
- `flowmatch/utils.py`: Utility functions for seeding, logging, EMA, and nearest-neighbor classification.
- `flowmatch/train_classifier.py`: Standalone script/module to train a ResNet classifier for evaluation purposes.
- `flowmatch/fid.py`: Script for calculating FID scores on trained models.

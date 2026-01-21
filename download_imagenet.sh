#!/bin/bash

huggingface-cli download --repo-type dataset cloneofsimo/imagenet.int8 --local-dir ./data/vae_mds
python -m xfm.dataset.filter_imagenet \
    --src ./data/vae_mds \
    --dst ./data/vae_mds_100 \
    --classes 100




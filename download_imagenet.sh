#!/bin/bash

huggingface-cli download --repo-type dataset cloneofsimo/imagenet.int8 --local-dir ./data/vae_mds
python flowmatch/dataset/filter_imagenet.py \
    --src ./data/vae_mds \
    --dst ./data/vae_mds_100 \
    --classes 100




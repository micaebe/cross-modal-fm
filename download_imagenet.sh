#!/bin/bash

huggingface-cli download --repo-type dataset cloneofsimo/imagenet.int8 --local-dir ./data/vae_mds
python -m xfm.dataset.filter_imagenet \
    --src ./data/vae_mds \
    --dst_train ./data/vae_mds_100/train \
    --dst_test ./data/vae_mds_100/test \
    --classes 100 \
    --test_samples 100 \
    --workers 4
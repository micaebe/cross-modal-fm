#!/bin/bash

COMMON_ARGS="--use_bf16 --epochs 101 --batch_size 256 --lr 5e-4 --device cuda --dataset cifar --project flowmatching_cifar_ablations4 --wandb --classifier_path classifier_cifar.pth"
run_exp() {
    echo "----------------------------------------------------------------"
    echo "Running experiment with args: $@"
    echo "----------------------------------------------------------------"
    
    python main.py $COMMON_ARGS "$@"
    
    if [ $? -ne 0 ]; then
        echo "Error: Previous run failed. Stopping script."
        exit 1
    fi
}

echo ">>> Rectangle 0.2"
run_exp --source image --target label --label_embedding "rectangle" --embedding_std_scale 0.2
run_exp --source label --target image --label_embedding "rectangle" --embedding_std_scale 0.2
run_exp --source image --target label --label_embedding "rectangle" --embedding_std_scale 0.2 --use_ln --ln_loc 0.5 --ln_scale 1.2
run_exp --source image --target label --label_embedding "rectangle" --embedding_std_scale 0.2 --use_ln --ln_loc 0.0 --ln_scale 1.0
run_exp --source image --target label --label_embedding "rectangle" --embedding_std_scale 0.2 --bidirectional
run_exp --source image --target label --label_embedding "rectangle" --embedding_std_scale 0.2 --bidirectional --use_ln --ln_loc 0.5 --ln_scale 1.2
echo ">>> Rectangle 0.5"
run_exp --source label --target image --label_embedding "rectangle" --embedding_std_scale 0.5
run_exp --source image --target label --label_embedding "rectangle" --embedding_std_scale 0.5
run_exp --source image --target label --label_embedding "rectangle" --embedding_std_scale 0.5 --bidirectional
echo ">>> Grayscale"
run_exp --source label --target image --label_embedding "grayscale" --embedding_std_scale 0.5
run_exp --source image --target label --label_embedding "grayscale" --embedding_std_scale 0.5
run_exp --source image --target label --label_embedding "grayscale" --embedding_std_scale 0.5 --bidirectional
echo ">>> Baseline"
run_exp --source noise --target image --use_conditioning
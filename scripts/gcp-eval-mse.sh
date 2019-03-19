#!/usr/bin/env bash

# Create images from test dataset for evaluation

cd ..

python train.py \
    --experiment-name=eval_mse_grayscale \
    --task=eval-mse \
    --eval-type=grayscale \
    --dataset-name=places100 \
    --val-batch-size=1 \
    --batch-output-frequency=25 \

python train.py \
    --experiment-name=eval_mse_resnet \
    --task=eval-mse \
    --eval-type=colorized \
    --model-name=resnet \
    --dataset-name=places100 \
    --val-batch-size=1 \
    --batch-output-frequency=25 \

# TODO: Add UNet and GAN
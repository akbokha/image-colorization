#!/usr/bin/env bash

cd ..

python train.py \
    --experiment-name=eval_ps_grayscale \
    --task=eval-ps \
    --eval-type=grayscale \
    --batch-output-frequency=1 \

python train.py \
    --experiment-name=eval_ps_resnet \
    --task=eval-ps \
    --eval-type=colorized \
    --model-name=resnet \
    --batch-output-frequency=1 \

# TODO: Add UNet and GAN
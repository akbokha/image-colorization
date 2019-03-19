#!/usr/bin/env bash

cd ..

python train.py \
    --experiment-name=eval_si_original \
    --task=eval-si \
    --eval-type=original \
    --val-batch-size=25 \
    --batch-output-frequency=1 \

python train.py \
    --experiment-name=eval_si_grayscale \
    --task=eval-si \
    --eval-type=grayscale \
    --val-batch-size=25 \
    --batch-output-frequency=1 \

python train.py \
    --experiment-name=eval_si_resnet \
    --task=eval-si \
    --eval-type=colorized \
    --model-name=resnet \
    --val-batch-size=25 \
    --batch-output-frequency=1 \

# TODO: Add UNet and GAN
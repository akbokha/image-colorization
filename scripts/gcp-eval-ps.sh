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
    --model-suffix=091 \
    --batch-output-frequency=1 \

python train.py \
    --experiment-name=eval_ps_unet \
    --task=eval-ps \
    --eval-type=colorized \
    --model-name=unet \
    --model-suffix=059 \
    --batch-output-frequency=1 \

python train.py \
    --experiment-name=eval_ps_cgan \
    --task=eval-ps \
    --eval-type=colorized \
    --model-name=cgan \
    --model-suffix=042 \
    --batch-output-frequency=1 \
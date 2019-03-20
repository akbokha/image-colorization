#!/usr/bin/env bash

cd ..

python train.py \
    --experiment-name=eval_ps_grayscale \
    --task=eval-ps \
    --eval-type=grayscale \
    --batch-output-frequency=1 \

python train.py \
    --experiment-name=eval_ps_resnet_quant \
    --task=eval-ps \
    --eval-type=colorized \
    --model-name=resnet \
    --model-suffix=091 \
    --batch-output-frequency=1 \

python train.py \
    --experiment-name=eval_ps_resnet_qual \
    --task=eval-ps \
    --eval-type=colorized \
    --model-name=resnet \
    --model-suffix=073 \
    --batch-output-frequency=1 \

python train.py \
    --experiment-name=eval_ps_unet_quant \
    --task=eval-ps \
    --eval-type=colorized \
    --model-name=unet \
    --model-suffix=087 \
    --batch-output-frequency=1 \

python train.py \
    --experiment-name=eval_ps_unet_qual \
    --task=eval-ps \
    --eval-type=colorized \
    --model-name=unet \
    --model-suffix=078 \
    --batch-output-frequency=1 \

python train.py \
    --experiment-name=eval_ps_cgan_quant \
    --task=eval-ps \
    --eval-type=colorized \
    --model-name=cgan \
    --model-suffix=042 \
    --batch-output-frequency=1 \

python train.py \
    --experiment-name=eval_ps_cgan_qual \
    --task=eval-ps \
    --eval-type=colorized \
    --model-name=cgan \
    --model-suffix=028 \
    --batch-output-frequency=1 \
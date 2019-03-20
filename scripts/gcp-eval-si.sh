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
    --experiment-name=eval_si_resnet_quant \
    --task=eval-si \
    --eval-type=colorized \
    --model-name=resnet \
    --model-suffix=091 \
    --val-batch-size=25 \
    --batch-output-frequency=1 \

python train.py \
    --experiment-name=eval_si_resnet_qual \
    --task=eval-si \
    --eval-type=colorized \
    --model-name=resnet \
    --model-suffix=073 \
    --val-batch-size=25 \
    --batch-output-frequency=1 \

python train.py \
    --experiment-name=eval_si_unet_quant \
    --task=eval-si \
    --eval-type=colorized \
    --model-name=unet \
    --model-suffix=087 \
    --val-batch-size=25 \
    --batch-output-frequency=1 \

python train.py \
    --experiment-name=eval_si_unet_qual \
    --task=eval-si \
    --eval-type=colorized \
    --model-name=unet \
    --model-suffix=078 \
    --val-batch-size=25 \
    --batch-output-frequency=1 \

python train.py \
    --experiment-name=eval_si_cgan_quant \
    --task=eval-si \
    --eval-type=colorized \
    --model-name=cgan \
    --model-suffix=042 \
    --val-batch-size=25 \
    --batch-output-frequency=1 \

python train.py \
    --experiment-name=eval_si_cgan_qual \
    --task=eval-si \
    --eval-type=colorized \
    --model-name=cgan \
    --model-suffix=028 \
    --val-batch-size=25 \
    --batch-output-frequency=1 \
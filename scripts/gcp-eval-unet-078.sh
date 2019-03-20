#!/usr/bin/env bash

cd ..

python train.py \
    --experiment-name=eval_gen_unet_qual \
    --task=eval-gen \
    --eval-type=colorized \
    --model-name=unet \
    --model-suffix=078 \
    --dataset-name=places100 \
    --train-batch-size=25 \
    --val-batch-size=25 \
    --batch-output-frequency=1

python train.py \
    --experiment-name=eval_ps_unet_qual \
    --task=eval-ps \
    --eval-type=colorized \
    --model-name=unet \
    --model-suffix=078 \
    --batch-output-frequency=1
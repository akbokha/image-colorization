#!/usr/bin/env bash

# Create images from test dataset for evaluation

cd ..

python train.py \
    --experiment-name=eval_gen_original \
    --task=eval-gen \
    --eval-type=original \
    --dataset-name=places100 \
    --train-batch-size=25 \
    --val-batch-size=25 \
    --batch-output-frequency=1 \

python train.py \
    --experiment-name=eval_gen_grayscale \
    --task=eval-gen \
    --eval-type=grayscale \
    --dataset-name=places100 \
    --train-batch-size=25 \
    --val-batch-size=25 \
    --batch-output-frequency=1 \

python train.py \
    --experiment-name=eval_gen_resnet \
    --task=eval-gen \
    --eval-type=colorized \
    --model-name=resnet \
    --model-suffix=073 \
    --dataset-name=places100 \
    --train-batch-size=25 \
    --val-batch-size=25 \
    --batch-output-frequency=1 \

# TODO: Add UNet and GAN
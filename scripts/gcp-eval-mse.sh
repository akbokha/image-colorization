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
    --experiment-name=eval_mse_resnet_quant \
    --task=eval-mse \
    --eval-type=colorized \
    --model-name=resnet \
    --model-suffix=091 \
    --dataset-name=places100 \
    --val-batch-size=1 \
    --batch-output-frequency=25 \

python train.py \
    --experiment-name=eval_mse_resnet_qual \
    --task=eval-mse \
    --eval-type=colorized \
    --model-name=resnet \
    --model-suffix=073 \
    --dataset-name=places100 \
    --val-batch-size=1 \
    --batch-output-frequency=25 \

python train.py \
    --experiment-name=eval_mse_unet_quant \
    --task=eval-mse \
    --eval-type=colorized \
    --model-name=unet \
    --model-suffix=087 \
    --dataset-name=places100 \
    --val-batch-size=1 \
    --batch-output-frequency=25 \

python train.py \
    --experiment-name=eval_mse_unet_qual \
    --task=eval-mse \
    --eval-type=colorized \
    --model-name=unet \
    --model-suffix=078 \
    --dataset-name=places100 \
    --val-batch-size=1 \
    --batch-output-frequency=25 \

python train.py \
    --experiment-name=eval_mse_cgan_quant \
    --task=eval-mse \
    --eval-type=colorized \
    --model-name=cgan \
    --model-suffix=042 \
    --dataset-name=places100 \
    --val-batch-size=1 \
    --batch-output-frequency=25 \

python train.py \
    --experiment-name=eval_mse_cgan_qual \
    --task=eval-mse \
    --eval-type=colorized \
    --model-name=cgan \
    --model-suffix=028 \
    --dataset-name=places100 \
    --val-batch-size=1 \
    --batch-output-frequency=25 \
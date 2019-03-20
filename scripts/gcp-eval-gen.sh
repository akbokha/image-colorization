#!/usr/bin/env bash

# Create images from test dataset for evaluation

cd ..

#python train.py \
#    --experiment-name=eval_gen_original \
#    --task=eval-gen \
#    --eval-type=original \
#    --dataset-name=places100 \
#    --train-batch-size=25 \
#    --val-batch-size=25 \
#    --batch-output-frequency=1 \
#
#python train.py \
#    --experiment-name=eval_gen_grayscale \
#    --task=eval-gen \
#    --eval-type=grayscale \
#    --dataset-name=places100 \
#    --train-batch-size=25 \
#    --val-batch-size=25 \
#    --batch-output-frequency=1 \

python train.py \
    --experiment-name=eval_gen_resnet_quant \
    --task=eval-gen \
    --eval-type=colorized \
    --model-name=resnet \
    --model-suffix=091 \
    --dataset-name=places100 \
    --train-batch-size=25 \
    --val-batch-size=25 \
    --batch-output-frequency=1 \

python train.py \
    --experiment-name=eval_gen_resnet_qual \
    --task=eval-gen \
    --eval-type=colorized \
    --model-name=resnet \
    --model-suffix=073 \
    --dataset-name=places100 \
    --train-batch-size=25 \
    --val-batch-size=25 \
    --batch-output-frequency=1 \

python train.py \
    --experiment-name=eval_gen_unet_quant \
    --task=eval-gen \
    --eval-type=colorized \
    --model-name=unet \
    --model-suffix=087 \
    --dataset-name=places100 \
    --train-batch-size=25 \
    --val-batch-size=25 \
    --batch-output-frequency=1 \

python train.py \
    --experiment-name=eval_gen_unet_qual \
    --task=eval-gen \
    --eval-type=colorized \
    --model-name=unet \
    --model-suffix=078 \
    --dataset-name=places100 \
    --train-batch-size=25 \
    --val-batch-size=25 \
    --batch-output-frequency=1 \

python train.py \
    --experiment-name=eval_gen_cgan_quant \
    --task=eval-gen \
    --eval-type=colorized \
    --model-name=cgan \
    --model-suffix=042 \
    --dataset-name=places100 \
    --train-batch-size=25 \
    --val-batch-size=25 \
    --batch-output-frequency=1 \

python train.py \
    --experiment-name=eval_gen_cgan_qual \
    --task=eval-gen \
    --eval-type=colorized \
    --model-name=cgan \
    --model-suffix=028 \
    --dataset-name=places100 \
    --train-batch-size=25 \
    --val-batch-size=25 \
    --batch-output-frequency=1 \
#!/usr/bin/env bash

cd ..

python train.py \
    --task=classifier \
    --experiment-name=cls_1 \
    --dataset-name=places100 \
    --use-dataset-archive=0 \
    --train-batch-size=64 \
    --val-batch-size=64 \
    --batch-output-frequency=50 \
    --max-epochs=100
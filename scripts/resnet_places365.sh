#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=LongJobs
#SBATCH --gres=gpu:4
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-80:00:00

export CUDA_HOME=/opt/cuda-9.0.176.1/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

# Set up scratch disk directory
mkdir -p /disk/scratch/${STUDENT_ID}
export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

# copy places365 folder structure to scratch disk
echo "Copying dataset."
export DATASET_DIR=${TMP}data
mkdir -p $DATASET_DIR
rsync -ua /home/${STUDENT_ID}/image-colorization/data/places365 $DATASET_DIR
echo "DATASET_DIR: $DATASET_DIR"


# Activate the relevant virtual environment:
source /home/${STUDENT_ID}/miniconda3/bin/activate mlp

# Run script
echo "Starting training."
python /home/${STUDENT_ID}/image-colorization/train.py \
    --task=colorizer \
    --experiment-name=resnet_places365_long_300e \
    --model-name=resnet \
    --dataset-root-path=$DATASET_DIR \
    --dataset-name=places365 \
    --model-path=/home/${STUDENT_ID}/models/ \
    --train-batch-size=100 \
    --val-batch-size=100 \
    --batch-output-frequency=10 \
    --max-images=500 \
    --max-epochs=300

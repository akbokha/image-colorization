import random
import sys
import time

import torch
import torch.nn as nn
import torch.optim

from .dataloaders import *
from .models import *
from .options import ModelOptions
from .colorizer import train_colorizer
from .classifier import train_classifier
from .utils import *

task_names = ['colorizer', 'classifier']
dataset_names = ['placeholder', 'cifar10', 'places100', 'places205', 'places365']
colorizer_model_names = ['resnet', 'unet32', 'unet224'm 'nazerigan32', 'nazerigan224']

def main(options):
    # initialize random seed
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    
    gpu_available = torch.cuda.is_available()
    print(gpu_available)

    # Create experiment output directory
    if not os.path.exists(options.experiment_output_path):
        os.makedirs(options.experiment_output_path)

    args = vars(options)
    print('\n------------ Environment -------------')
    print('GPU Available: {0}'.format(gpu_available))
    print('\n------------ Options -------------')
    with open(os.path.join(options.experiment_output_path, 'options.dat'), 'w') as f:
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
            f.write('%s: %s\n' % (str(k), str(v)))

    # Check if specified dataset is one that is supported by experimentation framework
    if options.dataset_name not in dataset_names:
        print('{} is not a valid dataset. The supported datasets are: {}'.format(options.dataset_name, dataset_names))
        clean_and_exit(options)

    # Check if specified task is one that is supported by experimentation framework
    if options.task not in task_names:
        print('{} is not a valid task. The supported tasks are: {}'.format(options.task, task_names))
        clean_and_exit(options)

    if options.task == 'colorizer':

        # Create data loaders
        if options.dataset_name == 'placeholder':
            train_loader, val_loader = get_placeholder_loaders(
                options.dataset_path, options.train_batch_size, options.val_batch_size)

        elif options.dataset_name == 'cifar10':
            train_loader, val_loader = get_cifar10_loaders(
                options.dataset_path, options.train_batch_size, options.val_batch_size)

        elif options.dataset_name == 'places100':
            train_loader, val_loader = get_places365_loaders(
                options.dataset_path, options.train_batch_size, options.val_batch_size)

        elif options.dataset_name == 'places205':
            train_loader, val_loader = get_places205_loaders(
                options.dataset_path, options.train_batch_size, options.val_batch_size)

        elif options.dataset_name == 'places365':
            train_loader, val_loader = get_places365_loaders(
                options.dataset_path, options.train_batch_size, options.val_batch_size)

        # Check if specified model is one that is supported by experimentation framework
        if options.model_name not in colorizer_model_names:
            print('{} is not a valid model. The supported models are: {}'.format(
                options.model_name, colorizer_model_names))
            clean_and_exit(options)

        train_colorizer(gpu_available, options, train_loader, val_loader)

    elif options.task == 'classifier':

        if options.dataset_name == 'placeholder':
            train_loader, val_loader = get_placeholder_loaders(
                options.dataset_path, options.train_batch_size, options.val_batch_size, for_classification=True)
            options.dataset_num_classes = 2

        elif options.dataset_name == 'places365':
            train_loader, val_loader = get_places365_loaders(
                    options.dataset_path, options.train_batch_size, options.val_batch_size, for_classification=True)
            options.dataset_num_classes = 365

        else:
            print("{} is not a valid dataset for classifier task".format(options.dataset_name))
            clean_and_exit(options)

        train_classifier(gpu_available, options, train_loader, val_loader)

def clean_and_exit(options):
    os.rmdir(options.experiment_output_path)
    sys.exit(1)


if __name__ == "__main__":
    main(ModelOptions().parse())

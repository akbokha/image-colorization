import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import time
from .options import ModelOptions
from .models import *
from .dataloaders import *
from .utils import *

def main(options):

    # initialize random seed
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)

    gpu_available = torch.cuda.is_available()

    # Create output directory
    if not os.path.exists(options.experiment_output_path):
        os.makedirs(options.experiment_output_path)

    # Create data loaders
    if options.dataset_name == 'placeholder':
        train_loader, val_loader = get_placeholder_loaders(options.dataset_path, options.batch_size)

    # Create model
    if options.model_name == 'resnet':
        model = ResNetColorizationNet()

    # Define Loss function and optimizer
    criterion = nn.MSELoss().cuda() if gpu_available else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    args = vars(options)
    print('\n------------ Environment -------------')
    print('GPU Available: {0}'.format(gpu_available))

    print('\n------------ Options -------------')
    with open(os.path.join(options.experiment_output_path, 'options.dat'), 'w') as f:
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
            f.write('%s: %s\n' % (str(k), str(v)))

    # train model
    epoch_stats = {"epoch": [], "train_time": [], "train_loss": [], 'val_loss': []}
    for epoch in range(options.max_epochs):

        train_time, train_loss = train_epoch(epoch, train_loader, model, criterion, optimizer, gpu_available, options)
        val_loss = validate_epoch(epoch, train_loader, model, criterion, True, gpu_available, options)

        # Save epoch stats
        epoch_stats['epoch'].append(epoch)
        epoch_stats['train_time'].append(train_time)
        epoch_stats['train_loss'].append(train_loss)
        epoch_stats['val_loss'].append(val_loss)
        save_stats(options.experiment_output_path, 'train_stats.csv', epoch_stats, epoch)


def train_epoch(epoch, train_loader, model, criterion, optimizer, gpu_available, options):
    '''Train model on data in train_loader'''

    print('Starting training epoch {}'.format(epoch))

    # Prepare value counters and timers
    batch_times, data_times, loss_values = AverageMeter(), AverageMeter(), AverageMeter()

    # Switch model to train mode
    model.train()

    # Train for single eopch
    start_time = time.time()
    for i, (input_gray, input_ab, target) in enumerate(train_loader):

        # Use GPU if available
        if gpu_available: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

        # Record time to load data (above)
        data_times.update(time.time() - start_time)
        start_time = time.time()

        # Run forward pass
        output_ab = model(input_gray)
        loss = criterion(output_ab, input_ab)

        # Record loss and measure accuracy
        loss_values.update(loss.item(), input_gray.size(0))

        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record time to do forward and backward passes
        batch_times.update(time.time() - start_time)
        start_time = time.time()

        # Print stats -- in the code below, val refers to value, not validation
        if i % options.batch_output_frequency == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_times.val:.3f} ({batch_times.avg:.3f})\t'
                  'Data {data_times.val:.3f} ({data_times.avg:.3f})\t'
                  'Loss {loss_values.val:.4f} ({loss_values.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_times=batch_times,
                data_times=data_times, loss_values=loss_values))

    print('Finished training epoch {}'.format(epoch))

    return batch_times.sum + data_times.sum, loss_values.avg


def validate_epoch(epoch, val_loader, model, criterion, save_images, gpu_available, options):
    '''Validate model on data in val_loader'''

    print('Starting validation.')

    # Create image output paths
    image_output_root_path = os.path.join(options.experiment_output_path, 'images', 'epoch-{}'.format(epoch))
    image_output_paths = {
        'grayscale': os.path.join(image_output_root_path, 'gray'),
        'colorized': os.path.join(image_output_root_path, 'colorized')
    }
    for image_path in image_output_paths.values():
        if not os.path.exists(image_path):
            os.makedirs(image_path)

    # Prepare value counters and timers
    batch_times, data_times, loss_values = AverageMeter(), AverageMeter(), AverageMeter()

    # Switch model to validation mode
    model.eval()

    # Run through validation set
    start_time = time.time()
    for i, (input_gray, input_ab, target) in enumerate(val_loader):

        # Use GPU if available
        if gpu_available: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

        # Record time to load data (above)
        data_times.update(time.time() - start_time)
        start_time = time.time()

        # Run forward pass
        output_ab = model(input_gray)
        loss = criterion(output_ab, input_ab)

        # Record loss and measure accuracy
        loss_values.update(loss.item(), input_gray.size(0))

        # Save images to file
        if save_images:
            for j in range(min(len(output_ab), 10)):  # save at most 10 images per batch
                gray_layer = input_gray[j].detach().cpu()
                ab_layers = output_ab[j].detach().cpu()
                save_name = 'img-{}.jpg'.format(i * val_loader.batch_size + j)
                save_colorized_images(gray_layer, ab_layers, save_paths=image_output_paths, save_name=save_name)

        # Record time to do forward passes and save images
        batch_times.update(time.time() - start_time)
        start_time = time.time()

        # Print model accuracy -- in the code below, val refers to both value and validation
        if i % options.batch_output_frequency == 0:
            print('Validate: [{0}/{1}]\t'
                  'Time {batch_times.val:.3f} ({batch_times.avg:.3f})\t'
                  'Loss {loss_values.val:.4f} ({loss_values.avg:.4f})\t'.format(
                i, len(val_loader), batch_times=batch_times, loss_values=loss_values))

    print('Finished validation.')

    return loss_values.avg


if __name__ == "__main__":
    main(ModelOptions().parse())
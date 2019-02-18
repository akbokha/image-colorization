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

    # Create output directories
    if not os.path.exists(options.experiment_output_path):
        os.makedirs(options.experiment_output_path)

    # Create data loaders
    if options.dataset_name == 'placeholder':
        train_loader, val_loader = get_placeholder_loaders(options.dataset_path, options.batch_size)

    # Create model
    if options.model_name == 'resnet':
        model = ResNetColorizeModel()

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
        val_loss = validate(train_loader, model, criterion, False, gpu_available, options)

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
    data_times = AverageMeter()
    batch_times = AverageMeter()
    loss_values = AverageMeter()

    # Switch model to train mode
    model.train()

    # Train for single eopch
    start_time = time.time()
    for i, (input_gray, input_ab, target) in enumerate(train_loader):

        # Use GPU if available
        input_gray_variable = Variable(input_gray).cuda() if gpu_available else Variable(input_gray)
        input_ab_variable = Variable(input_ab).cuda() if gpu_available else Variable(input_ab)
        target_variable = Variable(target).cuda() if gpu_available else Variable(target)

        # Record time to load data (above)
        data_times.update(time.time() - start_time)
        start_time = time.time()

        # Run forward pass
        output_ab = model(input_gray_variable)
        loss = criterion(output_ab, input_ab_variable)

        # Record loss and measure accuracy
        loss_values.update(loss.item(), input_gray.size(0))

        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record time to do forward and backward passes
        batch_times.update(time.time() - start_time)

        # Print stats -- in the code below, val refers to value, not validation
        if i % options.batch_output_frequency == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_times.val:.3f} ({batch_times.avg:.3f})\t'
                  'Data {data_times.val:.3f} ({data_times.avg:.3f})\t'
                  'Loss {loss_values.val:.4f} ({loss_values.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_times=batch_times,
                data_times=data_times, loss_values=loss_values))

    print('Finished training epoch {}'.format(epoch))

    return (batch_times.sum + data_times.sum, loss_values.avg)


def validate(val_loader, model, criterion, save_images, gpu_available, options):
    '''Validate model on data in val_loader'''

    print('Starting validation.')

    # Prepare value counters and timers
    data_times = AverageMeter()
    batch_times = AverageMeter()
    loss_values = AverageMeter()

    # Switch model to validation mode
    model.eval()

    # Run through validation set
    start_time = time.time()
    for i, (input_gray, input_ab, target) in enumerate(val_loader):

        # Use GPU if available
        target = target.cuda() if gpu_available else target
        input_gray_variable = Variable(input_gray, volatile=True).cuda() if gpu_available else Variable(input_gray,
                                                                                                  volatile=True)
        input_ab_variable = Variable(input_ab, volatile=True).cuda() if gpu_available else Variable(input_ab,
                                                                                                    volatile=True)
        target_variable = Variable(target, volatile=True).cuda() if gpu_available else Variable(target, volatile=True)

        # Record time to load data (above)
        data_times.update(time.time() - start_time)
        start_time = time.time()

        # Run forward pass
        output_ab = model(input_gray_variable)  # throw away class predictions
        loss = criterion(output_ab, input_ab_variable)  # check this!

        # Record loss and measure accuracy
        loss_values.update(loss.item(), input_gray.size(0))

        # Save images to file
        # TODO
        #if save_images:
        #    for j in range(len(output_ab)):
        #        save_path = {'grayscale': 'outputs/gray/', 'colorized': 'outputs/color/'}
        #        save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch)
        #        visualize_image(input_gray[j], ab_input=output_ab[j].data, show_image=False, save_path=save_path,
        #                        save_name=save_name)

        # Record time to do forward passes and save images
        batch_times.update(time.time() - start_time)

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
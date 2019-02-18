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
    if options.dataset_name == 'test':
        train_loader, val_loader = get_placeholder_loaders()

    # Create model
    if options.model_name == 'resnet':
        model = ResNetColorizeModel()

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
    model.train()
    epoch_stats = {"curr_epoch": [], "train_loss": [], "val_loss": []}
    for epoch in range(options.max_epochs):
        train_epoch(epoch, train_loader, model, criterion, optimizer, gpu_available)


def train_epoch(epoch, train_loader, model, criterion, optimizer, gpu_available):

    print('Starting training epoch {}'.format(epoch))

    # Prepare value counters and timers
    data_times = AverageMeter()
    batch_times = AverageMeter()
    losses = AverageMeter()

    # Switch model to train mode
    model.train()

    # Train for single eopch
    start_time = time.time()
    for i, (input_gray, input_ab, target) in enumerate(train_loader):
        print("Starting epoch ".format(epoch))

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
        losses.update(loss.data[0], input_gray.size(0))

        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record time to do forward and backward passes
        batch_times.update(time.time() - start_time)

        # Print model accuracy -- in the code below, val refers to value, not validation
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_times.val:.3f} ({batch_times.avg:.3f})\t'
                  'Data {data_times.val:.3f} ({data_times.avg:.3f})\t'
                  'Loss {losses.val:.4f} ({losses.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_times=batch_times,
                data_times=data_times, losses=losses))

    print('Finished training epoch {}'.format(epoch))


if __name__ == "__main__":
    main(ModelOptions().parse())
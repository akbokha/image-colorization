import os
import time
import torch
from torch import nn, optim
from torchvision import models

from .utils import *


def build_vgg16_model(model_root_path, num_classes):
    model_path = os.path.join(model_root_path, 'vgg16-397923af.pth')
    vgg16_model = models.vgg16()
    vgg16_model.load_state_dict(torch.load(model_path))

    # Freeze training for all conv layers
    for param in vgg16_model.features.parameters():
        param.require_grad = False

    # Alter classifier architecture
    num_features = vgg16_model.classifier[6].in_features
    features = list(vgg16_model.classifier.children())[:-1]  # Remove last layer
    features.extend([nn.Linear(num_features, num_classes)])  # Add our layer with 4 outputs
    vgg16_model.classifier = nn.Sequential(*features)  # Replace the model classifier

    return vgg16_model


def train_classifier(gpu_available, options, train_loader, val_loader):

    model = build_vgg16_model(options.model_path, options.dataset_num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Use GPU if available
    if gpu_available:
        model = model.cuda()

    epoch_stats = {"epoch": [], "train_time": [], "train_loss": [], 'val_loss': []}

    for epoch in range(options.max_epochs):
        train_time, train_loss, val_loss = train_val_epoch(
            epoch, train_loader, val_loader, criterion, model, optimizer, scheduler)
        save_epoch_stats(epoch, epoch_stats, train_time, train_loss, val_loss, options.experiment_output_path)
        save_model_state(epoch, model, optimizer, options.experiment_output_path)

    # TODO save best model weights


def train_val_epoch(epoch, train_loader, val_loader, criterion, model, optimizer, scheduler):

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            scheduler.step()
            model.train()  # Set model to training mode
            print('Starting training epoch {}'.format(epoch))

        else:
            model.eval()  # Set model to evaluate mode
            print('Starting validation epoch {}'.format(epoch))

        # Prepare value counters and timers
        batch_times, data_times, loss_values = AverageMeter(), AverageMeter(), AverageMeter()

        start_time = time.time()
        if phase == 'train':
            loader =  train_loader
        else:
            loader = val_loader

        for i, (inputs, labels) in enumerate(loader):

            # Use GPU if available
            if gpu_available:
                inputs, labels = inputs.cuda(), labels.cuda()

            # Record time to load data (above)
            data_times.update(time.time() - start_time)
            start_time = time.time()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
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
                    epoch, i + 1, len(train_loader), batch_times=batch_times,
                    data_times=data_times, loss_values=loss_values))

        if phase == 'train':
            print('Finished training epoch {}'.format(epoch))
            epoch_train_time = batch_times.sum + data_times.sum
            epoch_train_loss = loss_values.avg
        else:
            print('Finished validation epoch {}'.format(epoch))
            epoch_val_loss = loss_values.avg

    return epoch_train_time, epoch_train_loss, epoch_val_loss


def save_epoch_stats(epoch, epoch_stats, train_time, train_loss, val_loss, path):
    epoch_stats['epoch'].append(epoch)
    epoch_stats['train_time'].append(train_time)
    epoch_stats['train_loss'].append(train_loss)
    epoch_stats['val_loss'].append(val_loss)
    save_stats(path, 'train_stats.csv', epoch_stats, epoch)
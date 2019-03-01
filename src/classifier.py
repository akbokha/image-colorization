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
    features.extend([nn.Linear(num_features, num_classes)])
    vgg16_model.classifier = nn.Sequential(*features)  # Replace the model classifier

    model = models.vgg16(pretrained=False)
    model.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes)

    return vgg16_model


def train_classifier(gpu_available, options, train_loader, val_loader):

    model = build_vgg16_model(options.model_path, options.dataset_num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Use GPU if available
    if gpu_available:
        model = model.cuda()

    epoch_stats = {
        "epoch": [],
        "train_time": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(options.max_epochs):
        train_time, train_loss, train_acc, val_loss, val_acc = train_val_epoch(
            epoch, train_loader, val_loader, criterion, model, optimizer, scheduler, gpu_available, options)
        save_epoch_stats(
            epoch, epoch_stats, train_time, train_loss, train_acc, val_loss, val_acc, options.experiment_output_path)
        save_model_state(epoch, model, optimizer, options.experiment_output_path)


def train_val_epoch(epoch, train_loader, val_loader, criterion, model, optimizer, scheduler, gpu_available, options):

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
        batch_times, data_times = AverageMeter(), AverageMeter()
        loss_values, acc_rate = AverageMeter(), RateMeter()

        start_time = time.time()
        if phase == 'train':
            loader = train_loader
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
                correct = torch.sum(preds == labels.data).item()

                # Record loss and measure accuracy
                loss_values.update(loss.item(), inputs.size(0))
                acc_rate.update(correct, inputs.size(0))

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
                      'data {data_times.val:.3f} ({data_times.avg:.3f})\t'
                      'proc {batch_times.val:.3f} ({batch_times.avg:.3f})\t'
                      'loss {loss_values.val:.4f} ({loss_values.avg:.4f})\t'
                      'acc ({acc_rate.avg:.4f})'.format(
                    epoch, i + 1, len(train_loader), batch_times=batch_times,
                    data_times=data_times, loss_values=loss_values, acc_rate=acc_rate))

        if phase == 'train':
            print('Finished training epoch {}'.format(epoch))
            epoch_train_time = batch_times.sum + data_times.sum
            epoch_train_loss = loss_values.avg
            epoch_train_acc = acc_rate.avg
        else:
            print('Finished validation epoch {}'.format(epoch))
            epoch_val_loss = loss_values.avg
            epoch_val_acc = acc_rate.avg

    return epoch_train_time, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc


def save_epoch_stats(epoch, epoch_stats, train_time, train_loss, train_acc, val_loss, val_acc, path):
    epoch_stats['epoch'].append(epoch)
    epoch_stats['train_time'].append(train_time)
    epoch_stats['train_loss'].append(train_loss)
    epoch_stats['train_acc'].append(train_acc)
    epoch_stats['val_loss'].append(val_loss)
    epoch_stats['val_acc'].append(val_acc)
    save_stats(path, 'train_stats.csv', epoch_stats, epoch)
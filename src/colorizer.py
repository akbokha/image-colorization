import math
import time
import torch.optim

from .models import *
from .utils import *

def train_colorizer(gpu_available, options, train_loader, val_loader):

    # Create model
    if options.model_name == 'resnet':
        model = ResNetColorizationNet()
    if options.model_name == 'unet32':
        model = UNet32()
    if options.model_name == 'unet224':
        model = UNet224()

    # Make model use gpu if available
    if gpu_available:
        model = model.cuda()

    # Define loss function and optimizer
    criterion = nn.MSELoss().cuda() if gpu_available else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # train model
    epoch_stats = {"epoch": [], "train_time": [], "train_loss": [], 'val_loss': []}
    for epoch in range(options.max_epochs):
        train_time, train_loss = train_colorizer_epoch(epoch, train_loader, model, criterion, optimizer, gpu_available, options)
        val_loss = validate_colorizer_epoch(epoch, val_loader, model, criterion, True, gpu_available, options)
        save_epoch_stats(epoch, epoch_stats, train_time, train_loss, val_loss, options.experiment_output_path)
        save_model_state(options.experiment_output_path, epoch, model, optimizer)


def train_colorizer_epoch(epoch, train_loader, model, criterion, optimizer, gpu_available, options):
    """
    Train model on data in train_loader
    """

    print('Starting training epoch {}'.format(epoch))

    # Prepare value counters and timers
    batch_times, data_times, loss_values = AverageMeter(), AverageMeter(), AverageMeter()

    # Switch model to train mode
    model.train()

    # Train for single epoch
    start_time = time.time()
    for i, (input_gray, input_ab, img_original) in enumerate(train_loader):

        # Use GPU if available
        if gpu_available:
            input_gray, input_ab, img_original = input_gray.cuda(), input_ab.cuda(), img_original.cuda()

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
                epoch, i+ 1, len(train_loader), batch_times=batch_times,
                data_times=data_times, loss_values=loss_values))

    print('Finished training epoch {}'.format(epoch))

    return batch_times.sum + data_times.sum, loss_values.avg


def validate_colorizer_epoch(epoch, val_loader, model, criterion, save_images, gpu_available, options):
    """
    Validate model on data in val_loader
    """

    print('Starting validation epoch {}'.format(epoch))

    # Create image output paths
    image_output_paths = {
        'grayscale': os.path.join(options.experiment_output_path, 'images', 'gray'),
        'original': os.path.join(options.experiment_output_path, 'images', 'original'),
        'colorized': os.path.join(options.experiment_output_path, 'colorizations', 'epoch-{0:03d}'.format(epoch))
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
    num_images_saved = 0
    num_images_per_batch = math.ceil(max(options.max_images / len(val_loader), 1))
    for i, (input_gray, input_ab, img_original) in enumerate(val_loader):

        # Use GPU if available
        if gpu_available:
            input_gray, input_ab, img_original = input_gray.cuda(), input_ab.cuda(), img_original.cuda()

        # Record time to load data (above)
        data_times.update(time.time() - start_time)
        start_time = time.time()

        # Run forward pass
        with torch.no_grad():
            output_ab = model(input_gray)
            loss = criterion(output_ab, input_ab)

        # Record loss and measure accuracy
        loss_values.update(loss.item(), input_gray.size(0))

        # Save images to file

        if save_images and num_images_saved < options.max_images:
            for j in range(min(num_images_per_batch, options.max_images - num_images_saved)):
                gray_layer = input_gray[j].detach().cpu()
                ab_layers = output_ab[j].detach().cpu()
                save_name = 'img-{}.jpg'.format(i * val_loader.batch_size + j)
                # save gray-scale image and respective ground-truth images after first epoch
                if epoch == 0:
                    save_colorized_images(gray_layer, ab_layers, img_original[j],
                                          save_paths=image_output_paths, save_name=save_name, save_static_images=True)
                # save colorizations after every epoch
                save_colorized_images(gray_layer, ab_layers, img_original[j],
                                      save_paths=image_output_paths, save_name=save_name)
                num_images_saved += 1

        # Record time to do forward passes and save images
        batch_times.update(time.time() - start_time)
        start_time = time.time()

        # Print model accuracy -- in the code below, val refers to both value and validation
        if i % options.batch_output_frequency == 0:
            print('Validate: [{0}/{1}]\t'
                  'Time {batch_times.val:.3f} ({batch_times.avg:.3f})\t'
                  'Loss {loss_values.val:.4f} ({loss_values.avg:.4f})\t'.format(
                i + 1, len(val_loader), batch_times=batch_times, loss_values=loss_values))

    print('Finished validation.')

    return loss_values.avg


def save_epoch_stats(epoch, epoch_stats, train_time, train_loss, val_loss, path):
    epoch_stats['epoch'].append(epoch)
    epoch_stats['train_time'].append(train_time)
    epoch_stats['train_loss'].append(train_loss)
    epoch_stats['val_loss'].append(val_loss)
    save_stats(path, 'train_stats.csv', epoch_stats, epoch)

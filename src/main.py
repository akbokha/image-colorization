import random
import sys
import time

import torch
import torch.nn as nn
import torch.optim

from .dataloaders import *
from .models import *
from .options import ModelOptions
from .utils import *

dataset_names = ['placeholder', 'cifar10', 'places205', 'places365']
model_names = ['resnet', 'unet32', 'nazerigan32', 'nazerigan224']

def main(options):
    # initialize random seed
    random.seed(options.seed)
    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    
    gpu_available = torch.cuda.is_available()
    is_gan = False

    # Create output directory
    if not os.path.exists(options.experiment_output_path):
        os.makedirs(options.experiment_output_path)

    # Check if specified dataset is one that is supported by experimentation framework
    if options.dataset_name not in dataset_names:
        print('{} is not a valid dataset. The supported datasets are: {}'.format(options.dataset_name, dataset_names))
        clean_and_exit(options)

    # Create data loaders
    if options.dataset_name == 'placeholder':
        train_loader, val_loader = get_placeholder_loaders(options.dataset_path, options.train_batch_size, options.val_batch_size)
    elif options.dataset_name == 'cifar10':
        train_loader, val_loader = get_cifar10_loaders(options.dataset_path, options.train_batch_size, options.val_batch_size)
    elif options.dataset_name == 'places205':
        train_loader, val_loader = get_places205_loaders(options.dataset_path, options.train_batch_size, options.val_batch_size)
    elif options.dataset_name == 'places365':
        train_loader, val_loader = get_places365_loaders(options.dataset_path, options.train_batch_size, options.val_batch_size)

    # Check if specified model is one that is supported by experimentation framework
    if options.model_name not in model_names:
        print('{} is not a valid model. The supported models are: {}'.format(options.model_name, model_names))
        clean_and_exit(options)

            
    args = vars(options)
    print('\n------------ Environment -------------')
    print('GPU Available: {0}'.format(gpu_available))

    print('\n------------ Options -------------')
    with open(os.path.join(options.experiment_output_path, 'options.dat'), 'w') as f:
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
            f.write('%s: %s\n' % (str(k), str(v)))

    # Define Loss function and optimizer
    if is_gan == False:
        criterion = nn.MSELoss().cuda() if gpu_available else nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # train model
        epoch_stats = {"epoch": [], "train_time": [], "train_loss": [], 'val_loss': []}
        for epoch in range(options.max_epochs):
            train_time, train_loss = train_epoch(epoch, train_loader, model, criterion, optimizer, gpu_available, options)
            val_loss = validate_epoch(epoch, val_loader, model, criterion, True, gpu_available, options)
            state_epoch_stats(epoch, epoch_stats, train_loss, train_time, val_loss, options)
            save_model_state(epoch, model, optimizer, options)
        
    if is_gan == True:
        criterionBCE = nn.BCELoss().cuda() if gpu_available else nn.BCELoss()
        criterionMSE = nn.MSELoss().cuda() if gpu_available else nn.MSELoss()
        optimizerG = torch.optim.Adam(model.parameters(), betas=(0.5, 0.999), lr=0.0002)
        optimizerD = torch.optim.Adam(discriminator.parameters(), betas=(0.5, 0.999), lr=0.0002)
        
        # train model
        epoch_stats = {"epoch": [], "train_time": [], "train_loss": [], 'val_loss': []}
        for epoch in range(options.max_epochs):
            train_time, train_loss = train_gan_epoch(epoch, train_loader, model, discriminator, criterionBCE, optimizerG, optimizerD, gpu_available, options)
            val_loss = validate_gan_epoch(epoch, val_loader, model, discriminator, criterionMSE, True, gpu_available, options)
            state_epoch_stats(epoch, epoch_stats, train_loss, train_time, val_loss, options)
            #save_model_state(epoch, model, optimizer, options)

def train_epoch(epoch, train_loader, model, criterion, optimizer, gpu_available, options):
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
        if gpu_available: input_gray, input_ab, img_original = input_gray.cuda(), input_ab.cuda(), img_original.cuda()

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
                epoch, i+1, len(train_loader), batch_times=batch_times,
                data_times=data_times, loss_values=loss_values))

    print('Finished training epoch {}'.format(epoch))

    return batch_times.sum + data_times.sum, loss_values.avg


def validate_epoch(epoch, val_loader, model, criterion, save_images, gpu_available, options):
    """
    Validate model on data in val_loader
    """

    print('Starting validation.')

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
    
    num_images_saved = 0

    # Run through validation set
    start_time = time.time()
    for i, (input_gray, input_ab, img_original) in enumerate(val_loader):

        # Use GPU if available
        if gpu_available: input_gray, input_ab, img_original = input_gray.cuda(), input_ab.cuda(), img_original.cuda()

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
            for j in range(min(len(output_ab), options.max_images - num_images_saved)):
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
                i+1, len(val_loader), batch_times=batch_times, loss_values=loss_values))

    print('Finished validation.')

    return loss_values.avg

def train_gan_epoch(epoch, train_loader, generator, discriminator, criterion, optimizerG, optimizerD, gpu_available, options):
    """
    Train model on data in train_loader
    """

    print('Starting training epoch {}'.format(epoch))

    # Prepare value counters and timers
    batch_times, data_times, loss_D_values, loss_G_values = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    # Switch model to train mode
    generator.train()
    discriminator.train()
    
    # Establish convention for real and fake labels during training
    real_label = 0.9
    fake_label = 0

    # Train for single epoch
    start_time = time.time()
    for i, (input_gray, input_ab, img_original) in enumerate(train_loader):

        # Use GPU if available
        if gpu_available: input_gray, input_ab, img_original = input_gray.cuda(), input_ab.cuda(), img_original.cuda()

        # Record time to load data (above)
        data_times.update(time.time() - start_time)
        start_time = time.time()
        
        # Format batch
        batch_size = input_gray.size(0)
        real_labels = torch.full((batch_size,), real_label)
        fake_labels = torch.full((batch_size,), fake_label)
        
        if gpu_available:
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
        
        # Zero gradients
        optimizerG.zero_grad()
        optimizerD.zero_grad()
        
        # Train discriminator with all real batch
        output_real = discriminator(torch.cat((input_gray, input_ab), dim=1))
        loss_real = criterion(output_real, real_labels)
        loss_real.backward()
        
        # Generate fake batch
        fakes = generator(input_gray)
        
        # Train discriminator with all fake batch
        output_fake = discriminator(torch.cat((input_gray, fakes), dim=1))
        loss_fake = criterion(output_fake, fake_labels)
        loss_fake.backward(retain_graph=True)
        
        # Calculate full loss discriminator
        loss_discriminator = loss_real + loss_fake
        # Update discriminator weights
        optimizerD.step()
        
        # Calculate full loss generator
        loss_generator = criterion(output_fake, real_labels)
        loss_generator.backward()
        # Update generator weights
        optimizerG.step()
        
        
        # Record loss and measure accuracy
        loss_D_values.update(loss_discriminator.item(), input_gray.size(0))
        loss_G_values.update(loss_generator.item(), input_gray.size(0))

        # Record time to do forward and backward passes
        batch_times.update(time.time() - start_time)
        start_time = time.time()

        # Print stats -- in the code below, val refers to value, not validation
        if i % options.batch_output_frequency == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_times.val:.3f} ({batch_times.avg:.3f})\t'
                  'Data {data_times.val:.3f} ({data_times.avg:.3f})\t'
                  'Loss Generator {loss_G_values.val:.4f} ({loss_G_values.avg:.4f})\t'
                  'Loss Discriminator {loss_D_values.val:.5f} ({loss_D_values.avg:.5f})\t'.format(
                epoch, i+1, len(train_loader), batch_times=batch_times,
                data_times=data_times, loss_D_values=loss_D_values, loss_G_values=loss_G_values))

    print('Finished training epoch {}'.format(epoch))

    return batch_times.sum + data_times.sum, loss_G_values.avg


def validate_gan_epoch(epoch, val_loader, generator, discriminator, criterion, save_images, gpu_available, options):
    """
    Validate model on data in val_loader
    """

    print('Starting validation.')

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

    # Switch models to validation mode
    generator.eval()
    discriminator.eval()
    
    num_images_saved = 0

    # Run through validation set
    start_time = time.time()
    for i, (input_gray, input_ab, img_original) in enumerate(val_loader):

        # Use GPU if available
        if gpu_available: input_gray, input_ab, img_original = input_gray.cuda(), input_ab.cuda(), img_original.cuda()

        # Record time to load data (above)
        data_times.update(time.time() - start_time)
        start_time = time.time()

        # Run forward pass
        with torch.no_grad():
            output_ab = generator(input_gray)
            loss = criterion(output_ab, input_ab)

        # Record loss and measure accuracy
        loss_values.update(loss.item(), input_gray.size(0))

        # Save images to file
        if save_images and num_images_saved < options.max_images:
            for j in range(min(len(output_ab), options.max_images - num_images_saved)):
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
                i+1, len(val_loader), batch_times=batch_times, loss_values=loss_values))

    print('Finished validation.')

    return loss_values.avg

def state_epoch_stats(epoch, epoch_stats, train_loss, train_time, val_loss, options):
    epoch_stats['epoch'].append(epoch)
    epoch_stats['train_time'].append(train_time)
    epoch_stats['train_loss'].append(train_loss)
    epoch_stats['val_loss'].append(val_loss)
    save_stats(options.experiment_output_path, 'train_stats.csv', epoch_stats, epoch)


def save_model_state(epoch, model, optimizer, options):
    model_state_path = os.path.join(options.experiment_output_path, 'models', 'epoch-{0:03d}'.format(epoch))
    if not os.path.exists(model_state_path):
        os.makedirs(model_state_path)

    state_dict = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    torch.save(state_dict, os.path.join(model_state_path, 'state_dict'))


def clean_and_exit(options):
    os.rmdir(options.experiment_output_path)
    sys.exit(1)


if __name__ == "__main__":
    main(ModelOptions().parse())

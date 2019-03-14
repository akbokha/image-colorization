import math
import time
import torch.optim

from .models import *
from .utils import *

def train_colorizer(gpu_available, options, train_loader, val_loader):

    is_gan = False
    
    # Create model
    if options.model_name == 'resnet':
        model = ResNetColorizationNet()
    if options.model_name == 'unet32':
        model = UNet32()
    if options.model_name == 'unet224':
        model = UNet224()
    if options.model_name == 'nazerigan32':
        model = UNet32()
        discriminator = NazeriDiscriminator32()
        is_gan = True
    if options.model_name == 'nazerigan224':
        model = UNet224()
        discriminator = NazeriDiscriminator224()
        is_gan = True

    # Make model use gpu if available
    if gpu_available:
        model = model.cuda()
        if is_gan:
            discriminator = discriminator.cuda()

    # Define loss function and optimizer
    if is_gan == False:
        criterion = nn.MSELoss().cuda() if gpu_available else nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        # train model
        epoch_stats = {"epoch": [], "train_time": [], "train_loss": [], 'val_loss': []}
        for epoch in range(options.max_epochs):
            train_time, train_loss = train_colorizer_epoch(epoch, train_loader, model, criterion, optimizer, gpu_available, options)
            val_loss = validate_colorizer_epoch(epoch, val_loader, model, criterion, True, gpu_available, options)
            save_epoch_stats(epoch, epoch_stats, train_time, train_loss, val_loss, options.experiment_output_path)
            save_model_state(options.experiment_output_path, epoch, model, optimizer)
    
    else:
        criterionBCE = nn.BCEWithLogitsLoss().cuda() if gpu_available else nn.BCEWithLogitsLoss()
        criterionL1 = nn.L1Loss().cuda() if gpu_available else nn.L1Loss()
        L1 = 100
        criterionMSE = nn.MSELoss().cuda() if gpu_available else nn.MSELoss()
        optimizerG = torch.optim.Adam(model.parameters(), betas=(0.5, 0.999), lr=0.0002)
        optimizerD = torch.optim.Adam(discriminator.parameters(), betas=(0.5, 0.999), lr=0.0002)
        
        # train model
        epoch_stats = {"epoch": [], "train_time": [], "train_loss": [], 'val_loss': []}
        for epoch in range(options.max_epochs):
            train_time, train_loss = train_gan_colorizer_epoch(epoch, train_loader, model, discriminator, criterionBCE, criterionL1, L1, optimizerG, optimizerD, gpu_available, options)
            val_loss = validate_gan_colorizer_epoch(epoch, val_loader, model, discriminator, criterionMSE, True, gpu_available, options)
            state_epoch_stats(epoch, epoch_stats, train_loss, train_time, val_loss, options)
            save_model_state(epoch, model, optimizer, options)


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

def train_gan_colorizer_epoch(epoch, train_loader, generator, discriminator, criterionBCE, criterionL1, L1, optimizerG, optimizerD, gpu_available, options):
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
        loss_real = criterionBCE(output_real[:,0,0,0], real_labels)
        loss_real.backward()
        
        # Generate fake batch
        fakes = generator(input_gray)
        
        # Train discriminator with all fake batch
        output_fake = discriminator(torch.cat((input_gray, fakes), dim=1))
        loss_fake = criterionBCE(output_fake[:,0,0,0], fake_labels)
        loss_fake.backward(retain_graph=True)
        
        # Calculate full loss discriminator
        loss_discriminator = loss_real + loss_fake
        # Update discriminator weights
        optimizerD.step()
        
        # Calculate full loss generator
        loss_generator_BCE = criterionBCE(output_fake[:,0,0,0], real_labels)
        loss_generator_L1 = criterionL1(output_fake, real_labels) * L1
        loss_generator = loss_generator_BCE + loss_generator_L1
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


def validate_gan_colorizer_epoch(epoch, val_loader, generator, discriminator, criterion, save_images, gpu_available, options):
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


def save_epoch_stats(epoch, epoch_stats, train_time, train_loss, val_loss, path):
    epoch_stats['epoch'].append(epoch)
    epoch_stats['train_time'].append(train_time)
    epoch_stats['train_loss'].append(train_loss)
    epoch_stats['val_loss'].append(val_loss)
    save_stats(path, 'train_stats.csv', epoch_stats, epoch)

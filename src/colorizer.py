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
    if options.model_name == 'cgan':
        model_gen = ConvGenerator()
        model_dis = ConvDiscriminator()

    if options.model_name == 'cgan':
        if gpu_available:
            model_gen.cuda()
            model_dis.cuda()

        optimizer_gen = torch.optim.Adam(model_gen.parameters())
        optimizer_dis = torch.optim.Adam(model_dis.parameters())

        optimizers = {
            'generator': optimizer_gen,
            'discriminator': optimizer_dis
        }

        models = {
            'generator': model_gen,
            'discriminator': model_dis
        }

        criterion = nn.BCELoss().cuda() if gpu_available else nn.BCELoss()
        l1_loss = nn.L1Loss().cuda() if gpu_available else nn.L1Loss()

        l1_weight = 100

        # train model
        epoch_stats = {"epoch": [], "train_time": [],
                       "train_loss_D": [], "train_loss_D_real": [], "train_loss_D_generated": [],
                       "train_loss_G": [], "train_loss_G_GAN": [], "train_loss_G_generated": [],
                       "val_loss_D": [], "val_loss_G": []}
        for epoch in range(options.max_epochs):
            train_time, train_loss_G, train_loss_G_GAN, train_loss_G_gen, train_loss_D, train_loss_D_real, \
            train_loss_D_gen = train_GAN_colorizer_epoch(epoch, train_loader, model_gen, model_dis, criterion, l1_loss,
                                               l1_weight, optimizers, gpu_available, options)

            val_loss_G, val_loss_D = validate_GAN_colorizer_epoch(epoch, val_loader, model_gen, model_dis, criterion,
                                                                  l1_loss, l1_weight, True, gpu_available, options)
            save_epoch_stats_GAN(epoch, epoch_stats, train_loss_G, train_loss_G_GAN, train_loss_G_gen,
                                 train_loss_D, train_loss_D_real, train_loss_D_gen,
                                 val_loss_G, val_loss_D, train_time, options.experiment_output_path)
            save_model_state(epoch, models, optimizers, options.experiment_output_path)

    else:  # resnet or u-net
        # Make model use gpu if available
        if gpu_available:
            model = model.cuda()

        # Define Loss function and optimizer
        criterion = nn.MSELoss().cuda() if gpu_available else nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        # train model
        epoch_stats = {"epoch": [], "train_time": [], "train_loss": [], 'val_loss': []}
        for epoch in range(options.max_epochs):
            train_time, train_loss = train_colorizer_epoch(epoch, train_loader, model, criterion, optimizer,
                                                           gpu_available,
                                                           options)
            val_loss = validate_colorizer_epoch(epoch, val_loader, model, criterion, True, gpu_available, options)
            save_epoch_stats(epoch, epoch_stats, train_loss, train_time, val_loss, options.experiment_output_path)
            save_model_state(epoch, model, optimizer, options.experiment_output_path)


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
                epoch, i + 1, len(train_loader), batch_times=batch_times,
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

    num_images_saved = 0

    # Run through validation set
    start_time = time.time()
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
                i + 1, len(val_loader), batch_times=batch_times, loss_values=loss_values))

    print('Finished validation.')

    return loss_values.avg


def train_GAN_colorizer_epoch(epoch, train_loader, gen_model, dis_model, criterion, l1_loss, l1_weight, optimizers,
                              gpu_available, options):
    """
    Train generator and discriminator model on data in train_loader
    """

    print('Starting training epoch {}'.format(epoch))

    # Prepare value counters and timers
    batch_times, data_times = AverageMeter(), AverageMeter()
    loss_G, loss_G_GAN, loss_G_real = AverageMeter(), AverageMeter(), AverageMeter()
    loss_D, loss_D_gen, loss_D_real = AverageMeter(), AverageMeter(), AverageMeter()

    # Switch model to train mode
    gen_model.train()
    dis_model.train()

    # Labels that are used by the discriminator to classify real and generated samples
    REAL = 1
    GENERATED = 0

    # Train for single epoch
    start_time = time.time()
    for i, (input_gray, input_ab, img_original) in enumerate(train_loader):

        # Use GPU if available
        if gpu_available: input_gray, input_ab, img_original = input_gray.cuda(), input_ab.cuda(), img_original.cuda()

        # convert to FloatTensor since thnn_conv2d_forward is not implemented for type torch.ByteTensor
        target_img = img_original.type('torch.cuda.FloatTensor') if gpu_available else img_original.type('torch.FloatTensor')

        # convert to range [-1, 1]
        target_img = (target_img - 127.5) / 127.5

        # convert to range [0, 100]
        input_l = input_gray

        # Record time to load data (above)
        data_times.update(time.time() - start_time)
        start_time = time.time()

        """
        Update Discriminator Network
        1. Train with real examples
        2. Train with generated examples
        """
        # Train with real examples
        dis_model.zero_grad()
        output = dis_model(target_img)

        label = torch.FloatTensor(target_img.size(0)).fill_(REAL).cuda() if gpu_available else torch.FloatTensor(
            target_img.size(0)).fill_(REAL)

        dis_err_real = criterion(torch.squeeze(output), label)
        dis_err_real.backward()

        # Train with generated examples
        generated_examples = gen_model(input_l)
        output = dis_model(generated_examples.detach())
        label = label.fill_(GENERATED)  # replace with zeroes

        dis_err_gen = criterion(torch.squeeze(output), label)
        dis_err_gen.backward()

        dis_error = dis_err_real + dis_err_gen
        optimizers['discriminator'].step()

        """
        Update Generator Network
        """
        gen_model.zero_grad()
        label = label.fill_(REAL)  # replace with ones
        output = dis_model(generated_examples)
        gen_err_g = criterion(torch.squeeze(output), label)
        gen_err_loss = l1_loss(generated_examples.view(generated_examples.size(0), -1),
                               target_img.view(target_img.size(0), -1)) * l1_weight

        gen_err = gen_err_g + gen_err_loss
        gen_err.backward()
        optimizers['generator'].step()

        # Record time to do forward and backward passes
        batch_times.update(time.time() - start_time)
        start_time = time.time()

        loss_G.update(gen_err.item(), target_img.size(0))
        loss_G_GAN.update(gen_err.item(), target_img.size(0))
        loss_G_real.update(gen_err_loss.item(), target_img.size(0))

        loss_D.update(dis_error.item(), target_img.size(0))
        loss_D_gen.update(dis_err_gen.item(), target_img.size(0))
        loss_D_real.update(dis_err_real.item(), target_img.size(0))

        # Print stats -- in the code below, val refers to value, not validation
        if i % options.batch_output_frequency == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_times.val:.3f} ({batch_times.avg:.3f})\t'
                  'Data {data_times.val:.3f} ({data_times.avg:.3f})\t'
                  'loss_G {loss_G.val:.4f} ({loss_G.avg:.4f})\t'
                  'loss_G_GAN {loss_G_GAN.val:.4f} ({loss_G_GAN.avg:.4f})\t'
                  'loss_G_real {loss_G_real.val:.4f} ({loss_G_real.avg:.4f})\t'
                  'loss_D {loss_D.val:.4f} ({loss_D.avg:.4f})\t'
                  'loss_D_gen {loss_D_gen.val:.4f} ({loss_D_gen.avg:.4f})\t'
                  'loss_D_real {loss_D_real.val:.4f} ({loss_D_real.avg:.4f})\t'.format(
                epoch, i + 1, len(train_loader), batch_times=batch_times,
                data_times=data_times, loss_G=loss_G, loss_G_GAN=loss_G_GAN, loss_G_real=loss_G_real,
                loss_D=loss_D, loss_D_gen=loss_D_gen, loss_D_real=loss_D_real))

    print('Finished training epoch {}'.format(epoch))

    return batch_times.sum + data_times.sum, loss_G.avg, loss_G_GAN.avg, loss_G_real.avg, \
           loss_D.avg, loss_D_real.avg, loss_D_gen.avg


def validate_GAN_colorizer_epoch(epoch, val_loader, gen_model, dis_model, criterion, l1_loss, l1_weight, save_images,
                                 gpu_available, options):
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
    batch_times, data_times, loss_G, loss_D = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    # Switch model to validation mode
    gen_model.eval()
    dis_model.eval()

    # Labels that are used by the discriminator to classify real and generated samples
    REAL = 1
    GENERATED = 0

    num_images_saved = 0

    # Run through validation set
    start_time = time.time()
    for i, (input_gray, input_ab, img_original) in enumerate(val_loader):

        # Use GPU if available
        if gpu_available: input_gray, input_ab, img_original = input_gray.cuda(), input_ab.cuda(), img_original.cuda()

        # convert to FloatTensor since thnn_conv2d_forward is not implemented for type torch.ByteTensor
        target_img = img_original.type('torch.cuda.FloatTensor') if gpu_available else img_original.type('torch.FloatTensor')

        # convert to range [-1, 1]
        target_img = (target_img - 127.5) / 127.5

        # convert to range [0, 100]
        input_l = input_gray

        # Record time to load data (above)
        data_times.update(time.time() - start_time)
        start_time = time.time()

        # Run forward pass
        with torch.no_grad():
            """
            Discriminator Network
            1. Validate with real examples
            2. Validate with generated examples
            """
            # Validate with real examples
            output = dis_model(target_img)
            label = torch.FloatTensor(target_img.size(0)).fill_(REAL).cuda() if gpu_available else torch.FloatTensor(
                target_img.size(0)).fill_(REAL)
            dis_err_real = criterion(torch.squeeze(output), label)

            # Validate with generated examples
            generated = gen_model(input_l)
            label = label.fill_(GENERATED)  # replace with zeroes
            output = dis_model(generated.detach())
            dis_err_gen = criterion(torch.squeeze(output), label)

            dis_error = dis_err_real + dis_err_gen

            """
            Generator Network
            """
            output = dis_model(generated)
            label = label.fill_(REAL)  # replace with ones

            gen_err_g = criterion(torch.squeeze(output), label)
            gen_err_L1 = l1_loss(generated.view(generated.size(0), -1),
                                 target_img.view(target_img.size(0), -1)) * l1_weight

            gen_err = gen_err_g + gen_err_L1

        loss_G.update(gen_err.item(), target_img.size(0))
        loss_D.update(dis_error.item(), target_img.size(0))

        # Save images to file
        if save_images and num_images_saved < options.max_images:
            for j in range(min(len(output), options.max_images - num_images_saved)):
                gray_layer = input_gray[j].detach().cpu()
                save_name = 'img-{}.jpg'.format(i * val_loader.batch_size + j)
                # save gray-scale image and respective ground-truth images after first epoch
                if epoch == 0:
                    save_colorized_images(gray_layer, None, img_original[j],
                                          save_paths=image_output_paths, save_name=save_name, save_static_images=True)
                # save colorizations after every epoch
                save_colorized_images(gray_layer, None, img_original[j],
                                      save_paths=image_output_paths, save_name=save_name, gan_result=True,
                                      generated=generated[j])
                num_images_saved += 1

        # Record time to do forward passes and save images
        batch_times.update(time.time() - start_time)
        start_time = time.time()

        # Print model accuracy -- in the code below, val refers to both value and validation
        if i % options.batch_output_frequency == 0:
            print('Validate: [{0}/{1}]\t'
                  'Time {batch_times.val:.3f} ({batch_times.avg:.3f})\t'
                  'Loss_G {loss_G.val:.4f} ({loss_G.avg:.4f})\t'
                  'Loss_D {loss_D.val:.4f} ({loss_D.avg:.4f})\t'.format(
                i + 1, len(val_loader), batch_times=batch_times, loss_G=loss_G, loss_D=loss_D))

    print('Finished validation.')

    return loss_G.avg, loss_D.avg


def save_epoch_stats_GAN(epoch, epoch_stats, train_loss_G, train_loss_G_GAN, train_loss_G_gen, train_loss_D,
                         train_loss_D_real, train_loss_D_gen, val_loss_G, val_loss_D, train_time, path):
    epoch_stats['epoch'].append(epoch)
    epoch_stats['train_time'].append(train_time)
    epoch_stats['train_loss_G'].append(train_loss_G)
    epoch_stats['train_loss_G_GAN'].append(train_loss_G_GAN)
    epoch_stats['train_loss_G_generated'].append(train_loss_G_gen)
    epoch_stats['train_loss_D'].append(train_loss_D)
    epoch_stats['train_loss_D_real'].append(train_loss_D_real)
    epoch_stats['train_loss_D_generated'].append(train_loss_D_gen)
    epoch_stats['val_loss_G'].append(val_loss_G)
    epoch_stats['val_loss_D'].append(val_loss_D)
    save_stats(path, 'train_stats.csv', epoch_stats, epoch)


def save_epoch_stats(epoch, epoch_stats, train_time, train_loss, val_loss, path):
    epoch_stats['epoch'].append(epoch)
    epoch_stats['train_time'].append(train_time)
    epoch_stats['train_loss'].append(train_loss)
    epoch_stats['val_loss'].append(val_loss)
    save_stats(path, 'train_stats.csv', epoch_stats, epoch)

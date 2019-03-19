import os
import matplotlib.pyplot as plt
from .models import *
from .utils import *


def get_idx_to_label(dataset):
    """
    Get inverse list of index to label lookups
    """
    return {i[1]: i[0] for i in dataset.class_to_idx.items()}


def build_colorization_model(gpu_available, model_path, model_name):
    model_state_path = os.path.join(model_path,'{}.pth'.format(model_name))

    if gpu_available:
        model_state = torch.load(model_state_path)['model_state']
    else:
        model_state = torch.load(model_state_path, map_location='cpu')['model_state']

    if 'resnet' in model_name:
        model = ResNetColorizationNet()

    #elif 'unet224' in model_name:
    #    model = UNet224()

    elif 'cgan' in model_name:
        model = ConvGenerator()

    model.load_state_dict(model_state)

    # Use GPU if available
    if gpu_available:
        model = model.cuda()

    return model


def save_image(img_data, root_path, label, file_name):
    """
    Save transformed image to disk
    """
    img_path = os.path.join(root_path, label)
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    if len(img_data.shape) == 2:
        plt.imsave(arr=img_data, fname=os.path.join(img_path, file_name), cmap='gray')
    else:
        plt.imsave(arr=img_data, fname=os.path.join(img_path, file_name))


def generate_eval_set(gpu_available, options, test_loader):
    """
    Create images from test set for evaluation
    """
    class_idx_to_label = get_idx_to_label(test_loader.dataset)

    if options.eval_type == 'original':
        output_path = os.path.join(options.eval_root_path, options.eval_type, 'test')

        for i, (layers_grayscale, layers_ab, imgs_original, targets) in enumerate(test_loader):
            for j in range(imgs_original.shape[0]):
                img_data = imgs_original[j].detach().cpu().numpy().transpose(1, 2, 0)
                label = class_idx_to_label[targets[j].item()]
                file_name = 'img-{0:04d}.jpg'.format(i * test_loader.batch_size + j)
                save_image(img_data, output_path, label, file_name)

            print_progress(i, len(test_loader), options.batch_output_frequency)


    elif options.eval_type == 'grayscale':
        output_path = os.path.join(options.eval_root_path, options.eval_type ,'test')

        for i, (layers_grayscale, layers_ab, imgs_original, targets) in enumerate(test_loader):
            for j in range(layers_grayscale.shape[0]):
                img_data = layers_grayscale[j].detach().cpu().squeeze().numpy()
                label = class_idx_to_label[targets[j].item()]
                file_name = 'img-{0:04d}.jpg'.format(i * test_loader.batch_size + j)
                save_image(img_data, output_path, label, file_name)

            print_progress(i, len(test_loader), options.batch_output_frequency)

    elif options.eval_type == 'colorized':
        output_path = os.path.join(options.eval_root_path, options.full_model_name, 'test')
        model = build_colorization_model(gpu_available, options.model_path, options.full_model_name)

        for i, (layers_grayscale, layers_ab, imgs_original, targets) in enumerate(test_loader):

            # Use GPU if available
            if gpu_available:
                layers_grayscale, layers_ab, imgs_original = layers_grayscale.cuda(), layers_ab.cuda(), imgs_original.cuda()

            with torch.no_grad():
                output_ab = model(layers_grayscale)

            for j in range(layers_grayscale.shape[0]):
                img_data = combine_lab_image_layers(layers_grayscale[j], output_ab[j])
                label = class_idx_to_label[targets[j].item()]
                file_name = 'img-{0:04d}.jpg'.format(i * test_loader.batch_size + j)
                save_image(img_data, output_path, label, file_name)

            print_progress(i, len(test_loader), options.batch_output_frequency)


def print_progress(batch_index, num_batches, output_freq):
    if batch_index % output_freq == 0:
        print_ts('[{0}/{1}] Processing images'.format(batch_index + 1, num_batches))
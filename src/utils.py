import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.color import lab2rgb


class AverageMeter(object):
    '''An easy way to compute and store both average and current values'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RateMeter(object):
    '''An easy way to compute and store rates'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.successes = 0
        self.trials = 0

    def update(self, successes, trials):
        self.successes += successes
        self.trials += trials
        self.avg = self.successes / self.trials


def save_model_state(epoch, model, optimizer, path):
    model_state_path = os.path.join(path, 'models', 'epoch-{0:03d}'.format(epoch))
    if not os.path.exists(model_state_path):
        os.makedirs(model_state_path)

    if isinstance(model, dict):  # GAN model: two models (generator and discriminator) & their respective optimizers
        state_dict = {
            'epoch': epoch,
            'gen_model_state': model['generator'].state_dict(),
            'dis_model_state': model['discriminator'].state_dict(),
            'gen_optimizer_state': optimizer['generator'].state_dict(),
            'dis_optimizer_state': optimizer['discriminator'].state_dict()
        }
    else:
        state_dict = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

    torch.save(state_dict, os.path.join(model_state_path, 'state_dict'))


def save_stats(experiment_log_dir, filename, stats_dict, current_epoch, continue_from_mode=False, save_full_dict=True):
    """
    Saves the statistics in stats dict into a csv file. Using the keys as the header entries and the values as the
    columns of a particular header entry
    :param experiment_log_dir: the log folder dir filepath
    :param filename: the name of the csv file
    :param stats_dict: the stats dict containing the data to be saved
    :param current_epoch: the number of epochs since commencement of the current training session
    (i.e. if the experiment continued from 100 and this is epoch 105, then pass relative distance of 5.)
    :param save_full_dict: whether to save the full dict as is overriding any previous entries (might be useful if we
    want to overwrite a file)
    :return: The filepath to the summary file
    """

    summary_filename = os.path.join(experiment_log_dir, filename)
    mode = 'a' if continue_from_mode else 'w'
    with open(summary_filename, mode) as f:
        writer = csv.writer(f)
        if not continue_from_mode:
            writer.writerow(list(stats_dict.keys()))

        if save_full_dict:
            total_rows = len(list(stats_dict.values())[0])
            for idx in range(total_rows):
                row_to_add = [value[idx] for value in list(stats_dict.values())]
                writer.writerow(row_to_add)
        else:
            row_to_add = [value[current_epoch] for value in list(stats_dict.values())]
            writer.writerow(row_to_add)

    return summary_filename


def save_colorized_images(grayscale_layer, ab_layers, img_original, save_paths, save_name, save_static_images=False,
                          gan_result=False, generated=None):
    """
    Save grayscale and colorised versions of selected image
    """
    if save_static_images:  # save non-changing gray-scale and ground_truth images
        grayscale_input = grayscale_layer.squeeze().numpy()
        plt.imsave(arr=grayscale_input, fname=os.path.join(save_paths['grayscale'], save_name), cmap='gray')
        plt.imsave(arr=img_original.numpy().transpose((1, 2, 0)), fname=os.path.join(save_paths['original'], save_name))
    elif gan_result:
        generated = generated.numpy()
        generated = generated.transpose((1, 2, 0))  # rescale for matplotlib
        generated = (generated.astype(np.float64) + 1) / 2.

        plt.imsave(arr=generated, fname=os.path.join(save_paths['colorized'], save_name))
    else:  # save colorization results
        color_image = torch.cat((grayscale_layer, ab_layers), 0).numpy()  # combine channels
        color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
        color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
        color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
        color_image = lab2rgb(color_image.astype(np.float64))
        plt.imsave(arr=color_image, fname=os.path.join(save_paths['colorized'], save_name))

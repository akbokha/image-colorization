import os
import torch
from torch import nn
from torchvision import models
from .utils import *

def evaluate_mse(gpu_available, options, original_loader, eval_loader):

    criterion = nn.MSELoss().cuda() if gpu_available else nn.MSELoss()

    loss_stats = AverageMeter()
    original_loader_iter = iter(original_loader)
    eval_loader_iter = iter(eval_loader)
    for i in range(len(original_loader)):
        original_grayscale, original_ab, original_original, original_targets = next(original_loader_iter)
        eval_grayscale, eval_ab, eval_original, eval_targets = next(eval_loader_iter)

        ab_loss = criterion(original_ab, eval_ab)
        loss_stats.update(ab_loss.item(), original_ab.shape[0])

        if i % options.batch_output_frequency == 0:
            print_ts('[{0}/{1}]\tavg_loss {2:.5f}\tse_loss +/-{3:.5f}'.format(i + 1, len(original_loader), loss_stats.avg, loss_stats.se))

    output_path = options.experiment_output_path
    epoch_stats = { 'avg_mse': [loss_stats.avg], 'se_mse': [loss_stats.se]}
    save_stats(output_path, 'mse_loss.csv', epoch_stats, 1)
import os
import torch
from torch import nn
from torchvision import models
from .utils import *


def build_vgg16_places100_model(gpu_available, model_path):
    vgg16_model = models.vgg16()

    # Alter classifier architecture
    num_features = vgg16_model.classifier[6].in_features
    features = list(vgg16_model.classifier.children())[:-1]  # Remove last layer
    features.extend([nn.Linear(num_features, 100)]) # Replace last layer
    vgg16_model.classifier = nn.Sequential(*features)  # Replace the model classifier

    # Load trained weights
    model_path = os.path.join(model_path, 'vgg16-places100.pth')
    if gpu_available:
        model_state = torch.load(model_path)['model_state']
    else:
        model_state = torch.load(model_path, map_location='cpu')['model_state']
    vgg16_model.load_state_dict(model_state)
    return vgg16_model


def evaluate_si(gpu_available, options, test_loader):
    """
    Evaluate classification accuracy for test dataset
    """

    model = build_vgg16_places100_model(gpu_available, options.model_path)

    top1_acc, top5_acc = RateMeter(), RateMeter()
    for i, (inputs, targets) in enumerate(test_loader):
        outputs = model(inputs)
        acc = get_topk_correct(outputs, targets, ks=(1, 5))
        top1_acc.update(acc[0], inputs.size(0))
        top5_acc.update(acc[1], inputs.size(0))

        # Print stats -- in the code below, val refers to value, not validation
        if i % options.batch_output_frequency == 0:
            message = '[{0}/{1}]\t' \
                      'top1_acc {top1_acc.rate:.4f} ({top1_acc.avg_rate:.4f})\t' \
                      'top5_acc {top5_acc.rate:.4f} ({top5_acc.avg_rate:.4f})'.format(
                i + 1, len(test_loader), top1_acc=top1_acc, top5_acc=top5_acc)
            print_ts(message)

    output_path = options.experiment_output_path
    epoch_stats = { 'top1_acc': [top1_acc.avg_rate], 'top5_acc': [top5_acc.avg_rate] }
    save_stats(output_path, 'si_accuracy.csv', epoch_stats, 1)


def get_topk_correct(outputs, targets, ks=(1,)):
    """
    Computes the precision@K for the specified values of K
    """
    maxk = max(ks)
    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in ks:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.item())

    return res

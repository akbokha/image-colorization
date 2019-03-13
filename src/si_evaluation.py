import os
import torch
from torch import nn
from torchvision import models


def build_vgg16_places100_model(gpu_available, model_path):
    vgg16_model = models.vgg16()

    # Alter classifier architecture
    num_features = vgg16_model.classifier[6].in_features
    features = list(vgg16_model.classifier.children())[:-1]  # Remove last layer
    features.extend([nn.Linear(num_features, 100)]) # Replace last layer
    vgg16_model.classifier = nn.Sequential(*features)  # Replace the model classifier

    # Load trained weights
    model_path = os.path.join(model_path, 'vgg16_places100.pth')
    if gpu_available:
        model_state = torch.load(model_path)['model_state']
    else:
        model_state = torch.load(model_path, map_location='cpu')['model_state']
    vgg16_model.load_state_dict(model_state)
    return vgg16_model


def evaluate_si(gpu_available, options, test_loader):
    model = build_vgg16_places100_model(gpu_available, options.model_path)
    print(model)

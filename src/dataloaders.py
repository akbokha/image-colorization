import os
import pickle

import numpy as np
import torch
import torch.utils.data
from skimage.color import rgb2lab, rgb2gray
from torchvision import datasets, transforms


def get_placeholder_loaders(placeholder_path, batch_size):
    """
    Get placeholder data set loaders (for framework testing only)
    """

    train_directory = os.path.join(placeholder_path, 'train')
    train_transforms = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip()
    ])
    train_imagefolder = GrayscaleImageFolder(train_directory, train_transforms)
    train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=batch_size, shuffle=True,
                                               num_workers=1)

    val_transforms = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224)
    ])
    val_directory = os.path.join(placeholder_path, 'val')
    val_imagefolder = GrayscaleImageFolder(val_directory, val_transforms)
    val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=batch_size, shuffle=False,
                                             num_workers=1)

    return train_loader, val_loader


def unpickle_cifar10(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b"data"]


def get_cifar10_loaders(dataset_path, batch_size):
    """
    Get CIFAR-10 data set loaders
    """

    '''
    Process training data into a DataLoader object
    '''
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip()
    ])

    train_set = datasets.CIFAR10(root=dataset_path, train=True, download=True)
    num_training_points = train_set.__len__()
    num_points_training_batch = int(num_training_points / batch_size)

    train_data = np.array([]).reshape(0, 3, 32, 32)

    data_batch_name = 'cifar-10-batches-py/data_batch_{}'
    for batch_num in range(1, 6):
        data_batch = data_batch_name.format(batch_num)
        batch_dir = os.path.join(dataset_path, data_batch)
        train_data = np.append(train_data, np.reshape(unpickle_cifar10(batch_dir),
                                                      (num_points_training_batch, 3, 32, 32)), 0)

    train_lab_data = CIFAR10ImageDataSet(train_data, transforms=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_lab_data, batch_size=batch_size, shuffle=True, num_workers=1)

    '''
    Process validation data into a DataLoader object
    '''
    val_transforms = transforms.Compose([
        transforms.Scale(32)
    ])

    val_set_name = 'cifar-10-batches-py/test_batch'
    val_dir = os.path.join(dataset_path, val_set_name)
    val_data = unpickle_cifar10(val_dir)
    num_points_val_batch = val_data.shape[0]

    val_data = np.reshape(val_data, (num_points_val_batch, 3, 32, 32))

    val_lab_data = CIFAR10ImageDataSet(val_data, transforms=val_transforms)
    val_loader = torch.utils.data.DataLoader(val_lab_data, batch_size=1, shuffle=False, num_workers=1)

    return train_loader, val_loader


class GrayscaleImageFolder(datasets.ImageFolder):
    """
    Custom images folder, which converts images to grayscale before loading
    """

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img_original = self.transform(img)
            img_original = np.asarray(img_original)

            img_lab = rgb2lab(img_original)
            img_lab = (img_lab + 128) / 255

            img_ab = img_lab[:, :, 1:3]
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()

            img_gray = rgb2gray(img_original)
            img_gray = torch.from_numpy(img_gray).unsqueeze(0).float()

        return img_gray, img_ab, img_original


class CIFAR10ImageDataSet(torch.utils.data.Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms

    def __getitem__(self, index):
        img = self.data[index]

        img_original = transforms.functional.to_pil_image(torch.from_numpy(img.astype(np.uint8)))

        if self.transforms is not None:
            img_original = self.transforms(img_original)

        img_original = np.asarray(img_original)

        img_original = img_original / 255

        img_lab = rgb2lab(img_original)
        img_lab = (img_lab + 128) / 255

        img_ab = img_lab[:, :, 1:3]
        img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()

        img_gray = rgb2gray(img_original)
        img_gray = torch.from_numpy(img_gray).unsqueeze(0).float()

        return img_gray, img_ab, img_original

    def __len__(self):
        return self.data.shape[0]


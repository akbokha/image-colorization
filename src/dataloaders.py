import os
import pickle
import tarfile

import numpy as np
import torch
import torch.utils.data
from skimage.color import rgb2lab, rgb2gray
from torchvision import datasets, transforms


def files_are_present(dataset_path, num_files=None):
    if not os.path.exists(dataset_path):  # can be either a file or directory
        return False
    if num_files is not None:  # in case one wants to check the number of files in the directory
        files_present = [file for r, d, files in os.walk(dataset_path) for file in files if not file.startswith('.')]
        num_files_present = len(files_present)
        if num_files_present != num_files:  # to check whether the download + file-extraction was successful
            return False
    return True


def download_data(url, file, dataset_path, num_files=None):
    """
    :param url: the url of the data file
    :param file: the name of the data file (renaming allowed)
    :param dataset_path: the path where the data file should be
    :param num_files: the number of files that should be included in the data-file
    :return: whether data had to be downloaded (boolean can be used to do some post-processing)
    """
    if files_are_present(dataset_path, num_files):
        print("Files already downloaded")
        return False

    datasets.utils.download_url(url, root=dataset_path, filename=file, md5=None)

    with tarfile.open(os.path.join(dataset_path, file), 'r:gz') as tar:
        tar.extractall(path=dataset_path)

    return True


def unpickle_cifar10(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b"data"]


def get_224_train_transforms(for_classification=False):
    transform_list = [
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip()
    ]

    if for_classification:
        transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)


def get_224_val_transforms(for_classification=False):
    transform_list = [
        transforms.Scale(256),
        transforms.CenterCrop(224)
    ]

    if for_classification:
        transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)


def get_placeholder_loaders(dataset_path, train_batch_size, val_batch_size, for_classification=False):
    """
    Get placeholder data set loaders (for framework testing only)
    """

    train_directory = os.path.join(dataset_path, 'train')
    val_directory = os.path.join(dataset_path, 'val')

    train_transforms = get_224_train_transforms(for_classification)
    if for_classification:
        train_imagefolder = datasets.ImageFolder(train_directory, train_transforms)
    else:
        train_imagefolder = GrayscaleImageFolder(train_directory, train_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_imagefolder, batch_size=train_batch_size, shuffle=True, num_workers=1)

    val_transforms = get_224_val_transforms(for_classification)
    if for_classification:
        val_imagefolder = datasets.ImageFolder(val_directory, train_transforms)
    else:
        val_imagefolder = GrayscaleImageFolder(val_directory, val_transforms)

    val_loader = torch.utils.data.DataLoader(
        val_imagefolder, batch_size=val_batch_size, shuffle=False, num_workers=1)

    return train_loader, val_loader


def get_cifar10_loaders(dataset_path, train_batch_size, val_batch_size):
    """
    Get CIFAR-10 dataset loaders
    """

    # Process training data into a DataLoader object
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip()
    ])

    # Print if data is already downloaded
    data_batch_name = 'cifar-10-batches-py/data_batch_{}'
    for batch_num in range(1, 6):
        data_batch = data_batch_name.format(batch_num)
        data_file = os.path.join(dataset_path, data_batch)
        if os.path.exists(data_file):
            print("Batch {0} present".format(batch_num))

    # Used to download the data
    datasets.CIFAR10(root=dataset_path, train=True, download=True)

    train_data = np.array([]).reshape(0, 3, 32, 32)

    for batch_num in range(1, 6):
        data_batch = data_batch_name.format(batch_num)
        batch_dir = os.path.join(dataset_path, data_batch)
        train_data = np.append(train_data, np.reshape(unpickle_cifar10(batch_dir),
                                                      (10000, 3, 32, 32)), 0)

    train_lab_data = CIFAR10ImageDataSet(train_data, transforms=train_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_lab_data, batch_size=train_batch_size, shuffle=True, num_workers=1)

    # Process validation data into a DataLoader object
    val_transforms = transforms.Compose([
        transforms.Scale(32)
    ])

    val_set_name = 'cifar-10-batches-py/test_batch'
    val_dir = os.path.join(dataset_path, val_set_name)
    val_data = unpickle_cifar10(val_dir)
    num_points_val_batch = val_data.shape[0]

    val_data = np.reshape(val_data, (num_points_val_batch, 3, 32, 32))

    val_lab_data = CIFAR10ImageDataSet(val_data, transforms=val_transforms)
    val_loader = torch.utils.data.DataLoader(
        val_lab_data, batch_size=val_batch_size, shuffle=False, num_workers=1)

    return train_loader, val_loader


def get_places205_loaders(dataset_path, train_batch_size, val_batch_size):
    """
    Get Places205 dataset loaders
    """
    url = 'http://data.csail.mit.edu/places/places205/testSetPlaces205_resize.tar.gz'
    file_name = 'testSetPlaces205_resize.tar.gz'
    dir_name = 'testSet_resize'
    num_files = 41000

    validation_set_size = 1000

    train_directory = os.path.join(dataset_path, 'train')
    val_directory = os.path.join(dataset_path, 'val')

    had_to_download_data = download_data(url, file_name, dataset_path, num_files)

    if had_to_download_data:
        # need to place images in a sub-folder (see https://github.com/pytorch/examples/issues/236 for more info)
        val_img_dir = os.path.join(val_directory, 'class')
        train_img_dir = os.path.join(train_directory, 'class')
        os.makedirs(val_img_dir, exist_ok=True)
        os.makedirs(train_img_dir, exist_ok=True)

        full_dir = os.path.join(dataset_path, dir_name)
        for i, file in enumerate(os.listdir(full_dir)):
            if i < validation_set_size:  # first x will be val
                os.rename(os.path.join(full_dir, file), os.path.join(val_img_dir, file))
            else:  # others will be training
                os.rename(os.path.join(full_dir, file), os.path.join(train_img_dir, file))

        # remove the old directory
        os.rmdir(full_dir)

        # remove all the other obsolete/irrelevant files that have been downloaded
        to_be_removed_files = [file for file in os.listdir(dataset_path) if file not in ['train', 'val']]
        for file in to_be_removed_files:
            os.remove(os.path.join(dataset_path, file))

    train_transforms = get_224_train_transforms()
    train_imagefolder = GrayscaleImageFolder(train_directory, train_transforms)
    train_loader = torch.utils.data.DataLoader(
        train_imagefolder, batch_size=train_batch_size, shuffle=True, num_workers=1)

    val_transforms = get_224_val_transforms()
    val_imagefolder = GrayscaleImageFolder(val_directory, val_transforms)
    val_loader = torch.utils.data.DataLoader(
        val_imagefolder, batch_size=val_batch_size, shuffle=False, num_workers=1)

    return train_loader, val_loader


def get_places365_loaders(dataset_path, train_batch_size, val_batch_size, for_classification=False):
    """
    Get Places365 dataset loaders
    """

    train_directory = os.path.join(dataset_path, 'train')
    val_directory = os.path.join(dataset_path, 'val')

    train_transforms = get_224_train_transforms(for_classification)
    if for_classification:
        train_imagefolder = datasets.ImageFolder(train_directory, train_transforms)
    else:
        train_imagefolder = GrayscaleImageFolder(train_directory, train_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_imagefolder, batch_size=train_batch_size, shuffle=True, num_workers=1)

    val_transforms = get_224_val_transforms(for_classification)
    if for_classification:
        val_imagefolder = datasets.ImageFolder(val_directory, train_transforms)
    else:
        val_imagefolder = GrayscaleImageFolder(val_directory, val_transforms)

    val_loader = torch.utils.data.DataLoader(
        val_imagefolder, batch_size=val_batch_size, shuffle=False, num_workers=1)

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

            img_original = torch.from_numpy(img_original.transpose((2, 0, 1)))

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

import os
import numpy as np
from skimage.color import rgb2lab, rgb2gray, lab2rgb
import torch
import torch.utils.data
from torchvision import datasets, transforms

def get_placeholder_loaders(placeholder_path, batch_size):
    '''Get placeholder data set loaders (for framework testing only)'''

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
    val_imagefolder = GrayscaleImageFolder(val_directory , val_transforms)
    val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=batch_size, shuffle=False,
                                             num_workers=1)

    return train_loader, val_loader


class GrayscaleImageFolder(datasets.ImageFolder):
    '''Custom images folder, which converts images to grayscale before loading'''

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
            img_original = rgb2gray(img_original)
            img_original = torch.from_numpy(img_original).unsqueeze(0).float()
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_original, img_ab, target
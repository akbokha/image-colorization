import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import time
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage import io, color
import pickle
import copy

def unpickle(file):
    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b"data"]

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def load_data():
    # Load training data
    train = np.reshape(unpickle('cifar-10-batches-py/data_batch_1'), (10000, 3, 32, 32))
    train = np.append(train, np.reshape(unpickle('cifar-10-batches-py/data_batch_2'), (10000, 3, 32, 32)), 0)
    train = np.append(train, np.reshape(unpickle('cifar-10-batches-py/data_batch_3'), (10000, 3, 32, 32)), 0)
    train = np.append(train, np.reshape(unpickle('cifar-10-batches-py/data_batch_4'), (10000, 3, 32, 32)), 0)
    train = np.append(train, np.reshape(unpickle('cifar-10-batches-py/data_batch_5'), (10000, 3, 32, 32)), 0)

    # Convert to greyscale and LAB
    train_grey = np.zeros((50000, 1, 32, 32))
    train_lab  = np.zeros((50000, 3, 32, 32))
    for i in range(0, len(train)):
        grey = np.dot(train[i].transpose(1,2,0), [0.299, 0.587, 0.114])
        lab = color.rgb2lab(train[i].transpose(1,2,0)).transpose(2,0,1)
        train_grey[i][0] = grey
        train_lab[i] = lab
    # Convert to 0-1 range to avoid tanh fuckery
    train_lab = torch.tensor(train_lab/100).float()
    train_grey = torch.tensor(train_grey/255).float()


    # Load test and validation data
    testvalid = np.reshape(unpickle('cifar-10-batches-py/test_batch'), (10000, 3, 32, 32))
    valid = testvalid[0:9000]
    test = testvalid[0:1000]

    # Convert to greyscale and lab
    valid_grey = np.zeros((9000, 1, 32, 32))
    valid_lab = np.zeros((9000, 3, 32, 32))
    for i in range(0, len(test)):
        grey = np.dot(valid[i].transpose(1,2,0), [0.299, 0.587, 0.114])
        valid_grey[i][0] = grey

        lab = color.rgb2lab(valid[i].transpose(1,2,0)).transpose(2,0,1)
        valid_lab[i] = lab
    # Convert to 0-1 range to avoid tanh fuckery
    valid_lab = torch.tensor(valid_lab/100).float()
    valid_grey = torch.tensor(valid_grey/255).float()

    # Convert to greyscale
    test_grey = np.zeros((1000, 1, 32, 32))
    for i in range(0, len(test)):
        grey = np.dot(test[i].transpose(1,2,0), [0.299, 0.587, 0.114])
        test_grey[i][0] = grey
    # Convert to 0-1 range to avoid tanh fuckery
    test_grey = torch.tensor(test_grey/255).float()
    
    return train_lab, train_grey, valid_lab, valid_grey, test, test_grey

train_color, train_grey, valid_color, valid_grey, test_color, test_grey = load_data()

class CIFAR_iterator():
    def __init__(self, data_tuple, batch_size):
        self.data_tuple = data_tuple
        self.batch_size = batch_size
        self.i = 0
        self.iter = 0
        self.iters = np.floor_divide(data_tuple[0].size(0), batch_size)
        
    def getNext(self):            
        self.i += self.batch_size
        self.iter += 1
        res = (self.data_tuple[0][self.i:self.i +self.batch_size], self.data_tuple[1][self.i:self.i +self.batch_size])
        return res
    
    def getIter(self):
        return self.iter
    
    def getIters(self):
        return self.iters
    
    def reset(self):
        self.i = 0
        self.iter = 0
        
def to_rgb(grayscale_input, ab_input, save_path=None, save_name=None):
    '''Show/save rgb image from grayscale and ab channels
       Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}'''
    plt.clf()  # clear matplotlib
    color_image = ab_input.numpy()  # combine channels
    color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
    color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
    color_image = color.lab2rgb(color_image.astype(np.float64))
    grayscale_input = grayscale_input.squeeze().numpy()
    if save_path is not None and save_name is not None:
        plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
        plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))
        
use_gpu = torch.cuda.is_available()

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        #Convolution and deconvolution
        self.conv1 = nn.Conv2d(1, 48, (4, 4), stride=2, padding=1)
        self.conv2 = nn.Conv2d(48, 96, (4, 4), stride=2, padding=1)
        self.conv3 = nn.Conv2d(96, 192, (4, 4), stride=2, padding=1)
        self.conv4 = nn.Conv2d(192, 384, (4, 4), stride=2, padding=1)
        #self.conv5 = nn.Conv2d(3, 3, (4, 4), stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(384, 192, (4, 4), stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(384, 96, (4, 4), stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(192, 48, (4, 4), stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(96, 24, (4, 4), stride=2, padding=1)
        self.conv5 = nn.Conv2d(24, 3, (1, 1))
        
        #Batchnorm
        self.conv1_bnorm = nn.BatchNorm2d(48)
        self.conv2_bnorm = nn.BatchNorm2d(96)
        self.conv3_bnorm = nn.BatchNorm2d(192)
        self.conv4_bnorm = nn.BatchNorm2d(384)
        
        self.deconv1_bnorm = nn.BatchNorm2d(192)
        self.deconv2_bnorm = nn.BatchNorm2d(96)
        self.deconv3_bnorm = nn.BatchNorm2d(48)
        self.deconv4_bnorm = nn.BatchNorm2d(24)
    
    def forward(self, x32):
        # Contraction
        x16 = F.leaky_relu(self.conv1(x32), 0.2)
        x16 = self.conv1_bnorm(x16)
        
        x8 = F.leaky_relu(self.conv2(x16), 0.2)
        x8 = self.conv2_bnorm(x8)
        
        x4 = F.leaky_relu(self.conv3(x8), 0.2)
        x4 = self.conv3_bnorm(x4)
        
        x2 = F.leaky_relu(self.conv4(x4), 0.2)
        x2 = self.conv4_bnorm(x2)
        
        
        # Expansion
        x = F.relu(self.deconv1(x2))
        x = self.deconv1_bnorm(x)
        x4 = torch.cat((x,x4), 1)
        
        x = F.relu(self.deconv2(x4))
        x = self.deconv2_bnorm(x)
        x8 = torch.cat((x,x8), 1)
        
        x = F.relu(self.deconv3(x8))
        x = self.deconv3_bnorm(x)
        x16 = torch.cat((x,x16), 1)
        
        x = F.relu(self.deconv4(x16))
        x = self.deconv4_bnorm(x)
        
        # cross-channel parametric pooling
        # CHECK IF TANH IS A GOOD IDEA???
        x = F.tanh(self.conv5(x))
        #x = self.conv5(x)
        return x

# To track training loss
class AverageMeter(object):
    '''A handy class from the PyTorch ImageNet tutorial'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def write_results_to_file(file_dir, file_name, data):
    file = open(file_dir + os.path.sep + file_name, 'a')
    if isinstance(data, str):
        file.write(data + '\n')
    else:
        for line in data:
            file.write(line + '\n')
    file.close()

def train_model(train_loader, model, criterion, optimiser, epoch):
    print('Starting training epoch {}'.format(epoch))
    model.train()
    
    dir_name = "res"
    file_name = "intermediate"
    
    # Prepare value counters and timers
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    
    for i in range(0, train_loader.getIters()):
        
        (input_gray, target) = train_loader.getNext()
        # use gpu
        if use_gpu: input_gray, target = input_gray.cuda(), target.cuda()
        
        # Record load time data
        data_time.update(time.time() - end)
        
        # Run forward pass
        output_ab = model(input_gray)
        loss = criterion(output_ab, target)
        losses.update(loss.item(), input_gray.size(0))
        
        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
         # Record time to do forward and backward passes
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Print model accuracy -- in the code below, val refers to value, not validation
        if train_loader.getIter() % 25 == 0:
            stats = (
                'Epoch: [{0}][{1}/{2}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\tData {data_time.val:.3f} ({'
                'data_time.avg:.3f})\tLoss {loss.val:.4f} ({loss.avg:.4f})\t').format(
                epoch, i, train_loader.getIters(), batch_time=batch_time,
                data_time=data_time, loss=losses)
            print(stats)
            write_results_to_file(dir_name, file_name, stats)
            
    train_loader.reset()

    print('Finished training epoch {}'.format(epoch))
    
def validate(val_loader, model, criterion, save_images, epoch):
    model.eval()

    # Prepare value counters and timers
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    already_saved_images = False
    for i in range(0, val_loader.getIters()):
        data_time.update(time.time() - end)
        
        (input_gray, target) = val_loader.getNext()
        # use gpu
        if use_gpu: input_gray, target = input_gray.cuda(), target.cuda()

        # Run model and record loss
        output_ab = model(input_gray)  # throw away class predictions
        loss = criterion(output_ab, target)
        losses.update(loss.item(), input_gray.size(0))

        # Save images to file
        if save_images and not already_saved_images:
            already_saved_images = True
            for j in range(min(len(output_ab), 10)):  # save at most 5 images
                save_path = {'grayscale': 'res/grey/', 'colorized': 'res/color/'}
                save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch)
                to_rgb(input_gray[j].cpu(), ab_input=output_ab[j].detach().cpu(), save_path=save_path,
                       save_name=save_name)

        # Record time to do forward passes and save images
        batch_time.update(time.time() - end)
        end = time.time()

    val_loader.reset()
    print('Finished validation.')
    return losses.avg

# Load and process the inputs and targets
#targets, inputs, valid_targets, valid_inputs, test_inputs = load_data()
print("data  loaded")

# Make net
net = Unet()
print("net initialised")

# Ensure res directory exists
os.makedirs('res', exist_ok=True)
os.makedirs('res/grey', exist_ok=True)
os.makedirs('res/color/', exist_ok=True)

optimizer = optim.Adam(net.parameters(), lr=0.00002, betas=(0.5,0.999))
criterion = torch.nn.MSELoss()
epochs = 100
save_images = True

train_loader = CIFAR_iterator((train_grey, train_color), 64)
val_loader = CIFAR_iterator((valid_grey, valid_color), 64)

for epoch in range(0, epochs):
    train_model(train_loader, net, criterion, optimizer, epoch)
    with torch.no_grad():
        losses = validate(val_loader, net, criterion, save_images, epoch)
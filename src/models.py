import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNetColorizationNet(nn.Module):
    def __init__(self, input_size=128):
        super(ResNetColorizationNet, self).__init__()
        MIDLEVEL_FEATURE_SIZE = 128

        # First half: ResNet
        resnet = models.resnet18(num_classes=365)
        # Change first conv layer to accept single-channel (grayscale) input
        resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1))
        # Extract midlevel features from ResNet-gray
        self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])

        # Second half: Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(MIDLEVEL_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, input):
        # Pass input through ResNet-gray to extract features
        midlevel_features = self.midlevel_resnet(input)

        # Upsample to get colors
        output = self.upsample(midlevel_features)
        return output
    
class UNet32(nn.Module):
    def __init__(self):
        super(UNet32, self).__init__()
        #Convolution and deconvolution
        self.conv1 = nn.Conv2d(1, 64, (4, 4), stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, (4, 4), stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, (4, 4), stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, (4, 4), stride=2, padding=1)
        #self.conv5 = nn.Conv2d(3, 3, (4, 4), stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(512, 256, (4, 4), stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(512, 128, (4, 4), stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 64, (4, 4), stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, (4, 4), stride=2, padding=1)
        self.conv5 = nn.Conv2d(64, 2, (1, 1))
        
        #Batchnorm
        self.conv1_bnorm = nn.BatchNorm2d(64)
        self.conv2_bnorm = nn.BatchNorm2d(128)
        self.conv3_bnorm = nn.BatchNorm2d(256)
        self.conv4_bnorm = nn.BatchNorm2d(512)
        
        self.deconv1_bnorm = nn.BatchNorm2d(256)
        self.deconv2_bnorm = nn.BatchNorm2d(128)
        self.deconv3_bnorm = nn.BatchNorm2d(64)
        self.deconv4_bnorm = nn.BatchNorm2d(64)
    
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

class NazeriDiscriminator32(nn.Module):
    def __init__(self):
        super(NazeriDiscriminator32, self).__init__()
        #Convolution and deconvolution
        self.conv1 = nn.Conv2d(3, 64, (4, 4), stride=2, padding=1)     #32-16
        self.conv2 = nn.Conv2d(64, 128, (4, 4), stride=2, padding=1)   #16-8
        self.conv3 = nn.Conv2d(128, 256, (4, 4), stride=2, padding=1)  #8-4
        self.conv4 = nn.Conv2d(256, 512, (4, 4), stride=2, padding=1)  #4-2
        self.conv5 = nn.Conv2d(512, 1, (2, 2), stride=1, padding=0)    #2-1
        self.sigmoid = nn.Sigmoid()
        
        #Batchnorm
        self.conv1_bnorm = nn.BatchNorm2d(64)
        self.conv2_bnorm = nn.BatchNorm2d(128)
        self.conv3_bnorm = nn.BatchNorm2d(256)
        self.conv4_bnorm = nn.BatchNorm2d(512)
    
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
        
        x = self.conv5(x2)
        x = self.sigmoid(x)
        
        return x

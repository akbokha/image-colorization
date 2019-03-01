import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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
        # Convolution and deconvolution
        self.conv1 = nn.Conv2d(1, 64, (4, 4), stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, (4, 4), stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, (4, 4), stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, (4, 4), stride=2, padding=1)
        # self.conv5 = nn.Conv2d(3, 3, (4, 4), stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(512, 256, (4, 4), stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(512, 128, (4, 4), stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 64, (4, 4), stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, (4, 4), stride=2, padding=1)
        self.conv5 = nn.Conv2d(64, 2, (1, 1))

        # Batchnorm
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
        x4 = torch.cat((x, x4), 1)

        x = F.relu(self.deconv2(x4))
        x = self.deconv2_bnorm(x)
        x8 = torch.cat((x, x8), 1)

        x = F.relu(self.deconv3(x8))
        x = self.deconv3_bnorm(x)
        x16 = torch.cat((x, x16), 1)

        x = F.relu(self.deconv4(x16))
        x = self.deconv4_bnorm(x)

        # cross-channel parametric pooling
        x = F.tanh(self.conv5(x))
        # x = self.conv5(x)
        return x


class ConvGenerator(nn.Module):

    def __init__(self):
        super(ConvGenerator, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.1)

        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.LeakyReLU(0.1)

        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.LeakyReLU(0.1)

        self.conv5 = nn.Conv2d(512, 512, 3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.LeakyReLU(0.1)

        self.deconv6 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU()

        self.deconv7 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU()

        self.deconv8 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(128)
        self.relu8 = nn.ReLU()

        self.deconv9 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn9 = nn.BatchNorm2d(64)
        self.relu9 = nn.ReLU()

        self.deconv10 = nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn10 = nn.BatchNorm2d(3)
        self.relu10 = nn.ReLU()

        self._initialize_weights()

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu1(h)  # 64,112,112 (if input is 224x224)
        pool1 = h

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)  # 128,56,56
        pool2 = h

        h = self.conv3(h)  # 256,28,28
        h = self.bn3(h)
        h = self.relu3(h)
        pool3 = h

        h = self.conv4(h)  # 512,14,14
        h = self.bn4(h)
        h = self.relu4(h)
        pool4 = h

        h = self.conv5(h)  # 512,7,7
        h = self.bn5(h)
        h = self.relu5(h)

        h = self.deconv6(h)
        h = self.bn6(h)
        h = self.relu6(h)  # 512,14,14
        h += pool4

        h = self.deconv7(h)
        h = self.bn7(h)
        h = self.relu7(h)  # 256,28,28
        h += pool3

        h = self.deconv8(h)
        h = self.bn8(h)
        h = self.relu8(h)  # 128,56,56
        h += pool2

        h = self.deconv9(h)
        h = self.bn9(h)
        h = self.relu9(h)  # 64,112,112
        h += pool1

        h = self.deconv10(h)
        h = F.tanh(h)  # 3,224,224

        return h

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))


class ConvDiscriminator(nn.Module):

    def __init__(self):
        super(ConvDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.1)

        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.LeakyReLU(0.1)

        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.LeakyReLU(0.1)

        self.conv5 = nn.Conv2d(512, 512, 3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.LeakyReLU(0.1)

        self.conv6 = nn.Conv2d(512, 512, 7, stride=1, padding=0, bias=False)

        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.LeakyReLU(0.1)

        self.conv7 = nn.Conv2d(512, 1, 1, stride=1, padding=0, bias=False)

        self._initialize_weights()

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu1(h)  # 64,112,112 (if input is 224x224)

        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)  # 128,56,56

        h = self.conv3(h)  # 256,28,28
        h = self.bn3(h)
        h = self.relu3(h)

        h = self.conv4(h)  # 512,14,14
        h = self.bn4(h)
        h = self.relu4(h)

        h = self.conv5(h)  # 512,7,7
        h = self.bn5(h)
        h = self.relu5(h)

        h = self.conv6(h)
        h = self.bn6(h)
        h = self.relu6(h)  # 512,1,1

        h = self.conv7(h)
        h = F.sigmoid(h)

        return h

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

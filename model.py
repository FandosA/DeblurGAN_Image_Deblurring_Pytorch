# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:17:52 2025

@author: fandos
"""

import torch
import torch.nn as nn


def weights_init(m):
    
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
            
    elif classname.find('BatchNorm2d') != -1:
        
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)


class ResidualBlock(nn.Module):
    
    def __init__(self, features):
        
        super(ResidualBlock, self).__init__()
        
        self.reflection1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, bias=True)
        self.norm1 = nn.InstanceNorm2d(features, affine=False, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        
        self.dropout = nn.Dropout(0.5)
        
        self.reflection2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, bias=True)
        self.norm2 = nn.InstanceNorm2d(features, affine=False, track_running_stats=True)

    def forward(self, x):
        
        input_tensor = x
        
        x = self.relu(self.norm1(self.conv1(self.reflection1(x))))
        x = self.dropout(x)
        x = self.norm2(self.conv2(self.reflection2(x)))
        
        return input_tensor + x
    
    
class DeblurGAN(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=3, n_resblocks=9):
        
        super(DeblurGAN, self).__init__()
        
        # First convlution
        self.reflection1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, padding=0, bias=True)
        self.norm1 = nn.InstanceNorm2d(64, affine=False, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Second convlution
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True)
        self.norm2 = nn.InstanceNorm2d(128, affine=False, track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Third convlution
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True)
        self.norm3 = nn.InstanceNorm2d(256, affine=False, track_running_stats=True)
        self.relu3 = nn.ReLU(inplace=True)

        # Residual blocks
        res_blocks = []
        for _ in range(n_resblocks):
            res_blocks.append(ResidualBlock(256))
        self.res_blocks = nn.Sequential(*res_blocks)

        # First deconvolution
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.norm4 = nn.InstanceNorm2d(128, affine=False, track_running_stats=True)
        self.relu4 = nn.ReLU(inplace=True)

        # Second deconvolution
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        self.norm5 = nn.InstanceNorm2d(64, affine=False, track_running_stats=True)
        self.relu5 = nn.ReLU(inplace=True)

        # Last convolution
        self.reflection4 = nn.ReflectionPad2d(3)
        self.conv4 = nn.Conv2d(64, out_channels, kernel_size=7, padding=0, bias=True)
        self.tanh = nn.Tanh()
        
        # Initialize neural network weights
        self.apply(weights_init)

    def forward(self, x):
        
        input_tensor = x
        
        x = self.relu1(self.norm1(self.conv1(self.reflection1(x))))
        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))
        
        x = self.res_blocks(x)
        
        x = self.relu4(self.norm4(self.deconv1(x)))
        x = self.relu5(self.norm5(self.deconv2(x)))
        x = self.tanh(self.conv4(self.reflection4(x)))
        
        out = x + input_tensor
        out = torch.clamp(out, -1, 1)
        
        return out
    
    
class Discriminator(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=1):
        
        super(Discriminator, self).__init__()
        
        # First deconvolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=2, bias=True)
        self.leaky_relu1 = nn.LeakyReLU(0.2, True)
        
        # Second deconvolution
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=2, bias=True)
        self.norm2 = nn.InstanceNorm2d(128, affine=False, track_running_stats=True)
        self.leaky_relu2 = nn.LeakyReLU(0.2, True)
        
        # Third deconvolution
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=2, bias=True)
        self.norm3 = nn.InstanceNorm2d(256, affine=False, track_running_stats=True)
        self.leaky_relu3 = nn.LeakyReLU(0.2, True)
        
        # Fourth convolution
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=2, bias=True)
        self.norm4 = nn.InstanceNorm2d(512, affine=False, track_running_stats=True)
        self.leaky_relu4 = nn.LeakyReLU(0.2, True)
        
        # Last convolution
        self.conv5 = nn.Conv2d(512, out_channels, kernel_size=4, stride=1, padding=2, bias=True)
        
        # Initialize neural network weights
        self.apply(weights_init)

    def forward(self, x):
        
        x = self.leaky_relu1(self.conv1(x))
        
        x = self.leaky_relu2(self.norm2(self.conv2(x)))
        x = self.leaky_relu3(self.norm3(self.conv3(x)))
        x = self.leaky_relu4(self.norm4(self.conv4(x)))
        
        out = self.conv5(x)
        
        return out
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 15:22:22 2025

@author: fandos
"""

import torch
import torch.nn as nn
from torch.autograd import grad
from torchvision.models import vgg19, VGG19_Weights
from torchvision.transforms import Normalize
    
    
class GeneratorLoss(nn.Module):
    
    def __init__(self, lambda_perceptual=100, device=torch.device("cpu")):
        
        super(GeneratorLoss, self).__init__()
        
        vgg = vgg19(weights=VGG19_Weights.DEFAULT)
        
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:15]).eval().to(device)
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.mse_loss = nn.MSELoss()
        self.LAMBDA = lambda_perceptual

    def forward(self, image_fake, image_real, fake_logits):
        
        image_fake = (image_fake + 1) / 2
        image_real = (image_real + 1) / 2
        
        image_fake = self.normalize(image_fake)
        image_real = self.normalize(image_real)
        
        fake_features = self.feature_extractor(image_fake)
        real_features = self.feature_extractor(image_real).detach()
        
        perceptual_loss = self.mse_loss(fake_features, real_features)
        adversarial_loss = -fake_logits.mean()
        
        return self.LAMBDA * perceptual_loss + adversarial_loss
    
    
class DiscriminatorLossWGANGP(nn.Module):
    
    def __init__(self, lambda_gp=10, device=torch.device("cpu")):
        
        super(DiscriminatorLossWGANGP, self).__init__()
        
        self.LAMBDA = lambda_gp
        self.device = device

    def calc_gradient_penalty(self, discriminator, real_images, fake_images):
        
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_images.size())
        alpha = alpha.to(self.device)
        
        interpolates = alpha * real_images + (1 - alpha) * fake_images
        interpolates.requires_grad_(True)

        disc_interpolates = discriminator(interpolates)

        gradients = grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty

    def forward(self, discriminator, real_logits, fake_logits, real_images, fake_images):
        
        loss = fake_logits.mean() - real_logits.mean()
        gradient_penalty = self.calc_gradient_penalty(discriminator, real_images, fake_images)

        return loss + self.LAMBDA * gradient_penalty
    
    
class TVLoss(nn.Module):
    
    def __init__(self):
        
        super(TVLoss, self).__init__()

    def forward(self, image_generated):
        
        horizontal_diff = torch.abs(image_generated[:, :, 1:, :] - image_generated[:, :, :-1, :])
        vertical_diff = torch.abs(image_generated[:, :, :, 1:] - image_generated[:, :, :, :-1])
        
        tv_loss = torch.sum(horizontal_diff) + torch.sum(vertical_diff)
        
        return tv_loss

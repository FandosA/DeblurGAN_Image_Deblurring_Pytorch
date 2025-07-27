# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:26:05 2025

@author: fandos
"""

import os
import cv2
import random
import math
import utils
import torch
import numpy as np
import configargparse
import torch.nn as nn
from dataset import Dataset
import matplotlib.pyplot as plt
from model import DeblurGAN, Discriminator
from loss import DiscriminatorLossWGANGP
from torch.autograd import grad
from torch.autograd import Variable
import torchvision.models as models


def plotLossDiscriminator(log_dir, train_losses=None, img_name="img"):
    """
    If train_losses and validation_losses are None means that there are txt files
    with the loss values already saved in the folder, so they are loaded and the
    loss curves are shown in a plot and the image is saved too. If not, the function
    plots and saves the loss curves in a png file once the training is finished.
    :param log_dir: name of the folder to store the image or to load the loss values and plot the image
    :param train_losses: array with the loss values during the training
    :param val_losses: array with the loss values during the validation
    :return: 
    """
    if train_losses is None:
    
        files_in_dir = os.listdir(log_dir)
        
        for file in files_in_dir:
            
            if file == "train_losses_g.txt":
                train_losses = np.loadtxt(os.path.join(log_dir, "train_losses_g.txt"))
            
    epochs = np.arange(train_losses.shape[0])
    bestEpoch = np.argmin(train_losses)
    
    plt.figure()
    plt.plot(epochs, train_losses, label="Training loss", c='b')
    plt.plot(bestEpoch, train_losses[bestEpoch], label="Best epoch", c='y', marker='.', markersize=10)
    plt.text(bestEpoch+7, train_losses[bestEpoch]-0.15, str(bestEpoch), fontsize=8)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Discriminator loss along epochs')
    plt.legend()
    plt.draw()
    plt.savefig(os.path.join(log_dir, img_name + '.png'))


def plotLoss(log_dir, train_losses=None, validation_losses=None, img_name="img"):
    """
    If train_losses and validation_losses are None means that there are txt files
    with the loss values already saved in the folder, so they are loaded and the
    loss curves are shown in a plot and the image is saved too. If not, the function
    plots and saves the loss curves in a png file once the training is finished.
    :param log_dir: name of the folder to store the image or to load the loss values and plot the image
    :param train_losses: array with the loss values during the training
    :param val_losses: array with the loss values during the validation
    :return: 
    """
    if train_losses is None and validation_losses is None:
    
        files_in_dir = os.listdir(log_dir)
        
        for file in files_in_dir:
            
            if file == "train_losses_g.txt":
                train_losses = np.loadtxt(os.path.join(log_dir, "train_losses_g.txt"))
            elif file == "val_losses_g.txt":
                validation_losses = np.loadtxt(os.path.join(log_dir, "val_losses_g.txt"))
            
    epochs = np.arange(train_losses.shape[0])
    bestEpoch = np.argmin(validation_losses)
    
    plt.figure()
    plt.plot(epochs, train_losses, label="Training loss", c='b')
    plt.plot(epochs, validation_losses, label="Validation loss", c='r')
    plt.plot(bestEpoch, validation_losses[bestEpoch], label="Best epoch", c='y', marker='.', markersize=10)
    plt.text(bestEpoch+7, validation_losses[bestEpoch]-0.15, str(bestEpoch), fontsize=8)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Generator Loss along epochs')
    plt.legend()
    plt.draw()
    plt.savefig(os.path.join(log_dir, img_name + '.png'))

    
def plotAccuracy(log_dir, train_accs=None, validation_accs=None, img_name="img"):
    """
    If train_accuracies and validation_accuracies are None means that there are txt files
    with the accuracy values already saved in the folder, so they are loaded and the
    accuracy curves are shown in a plot and the image is saved too. If not, the function
    plots and saves the accuracy curves in a png file once the training is finished.
    :param log_dir: name of the folder to store the image or to load the accuracy values and plot the image
    :param train_accs: array with the accuracy values during the training
    :param val_accs: array with the accuracy values during the validation
    :return: 
    """
    if train_accs is None and validation_accs is None:
    
        files_in_dir = os.listdir(log_dir)
        
        for file in files_in_dir:
            
            if file == "train_accs_d.txt":
                train_accs = np.loadtxt(os.path.join(log_dir, "train_accs_d.txt"))
            elif file == "val_accs_d.txt":
                validation_accs = np.loadtxt(os.path.join(log_dir, "val_accs_d.txt"))
            
    epochs = np.arange(train_accs.shape[0])
    bestEpoch = np.argmax(validation_accs)
    
    plt.figure()
    plt.plot(epochs, train_accs, label="Training accuracy", c='b')
    plt.plot(epochs, validation_accs, label="Validation accuracy", c='r')
    plt.plot(bestEpoch, validation_accs[bestEpoch], label="Best epoch", c='y', marker='.', markersize=10)
    plt.text(bestEpoch+7, validation_accs[bestEpoch]-0.15, str(bestEpoch), fontsize=8)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Discriminator accuracy along epochs')
    plt.legend()
    plt.draw()
    plt.savefig(os.path.join(log_dir, img_name + '.png'))


if __name__ == "__main__":
    
    plotAccuracy("deblurGAN_bs1_lr0.0001_numresblocks9",
                       np.loadtxt("deblurGAN_bs1_lr0.0001_numresblocks9/train_accs_d.txt"),
                       np.loadtxt("deblurGAN_bs1_lr0.0001_numresblocks9/val_accs_d.txt"),
                       "discriminator_acc")
    
    plotLossDiscriminator("deblurGAN_bs1_lr0.0001_numresblocks9",
                       np.loadtxt("deblurGAN_bs1_lr0.0001_numresblocks9/train_losses_d.txt"),
                       "discriminator_loss")
    
    plotLoss("deblurGAN_bs1_lr0.0001_numresblocks9",
                       np.loadtxt("deblurGAN_bs1_lr0.0001_numresblocks9/train_losses_g.txt"),
                       np.loadtxt("deblurGAN_bs1_lr0.0001_numresblocks9/val_losses_g.txt"),
                       "generator_loss")
    
    plt.show()
    
    # img_path = "../OpenCV contest/dataset/TartanAir dataset train/endofworld/image_left/000445_left.png"
    
    # image_hr = cv2.imread(img_path)
    # image_hr = cv2.cvtColor(image_hr, cv2.COLOR_BGR2RGB)

    # # Generar un valor aleatorio de sigma (borrosidad)
    # kernel_size = random.choice([3, 5, 7])  # debe ser impar
    # sigma = random.uniform(0.5, 5.0)

    # image_blurred = cv2.GaussianBlur(image_hr, (kernel_size, kernel_size), sigmaX=sigma)

    # # Mostrar lado a lado
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # axs[0].imshow(image_hr)
    # axs[0].set_title("Original")
    # axs[0].axis('off')

    # axs[1].imshow(image_blurred)
    # axs[1].set_title(f"Blurred\nkernel={kernel_size}, sigma={sigma:.2f}")
    # axs[1].axis('off')

    # plt.tight_layout()
    # plt.show()
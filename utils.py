# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 17:38:15 2023

@author: andre
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import transforms


def selectDevice(args):
    """
    Select the device on which the model will run
    :param args: parameters of the project
    :return: device on which the model will run
    """
    if args.GPU and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('Device assigned: GPU (' + torch.cuda.get_device_name(device) + ')\n')
    else:
        device = torch.device("cpu")
        if args.GPU and not torch.cuda.is_available():
            print('GPU not available, device assigned: CPU\n')
        else:
            print('Device assigned: CPU\n')
            
    return device
    

def calculateAccuracy(logits, labels):
    """
    Compute discriminator accuracy
    :param logits (torch.Tensor): discriminator outputs before applying activation function BCEWithLogits
    :param labels (torch.Tensor): Etiquetas reales (1 para real, 0 para fake)
    :return: accuracy in X% format
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    correct = (preds == labels).float().sum()
    accuracy = correct / labels.numel()
    
    return accuracy


def tensorToImage(tensor):
    """
    Function to convert the pytorch tensor returned by the generator in an image array
    :param tensor_image (torch.Tensor): output tensor of the generator, an image normalized between -1 and 1
    :return: image in numpy array format, normlized between integer values 0 and 255 
    """
    tensor = tensor * 0.5 + 0.5
    image = tensor.squeeze().detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image


def imageToTensor(image):
    """
    Function to convert the image to a pytorch tensor in order to introduce it to the generator
    :param image: image to convert to a tensor
    :return: tensor_image (torch.Tensor) normalized between -1 and 1
    """
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transformer(image)
    
    return image

    
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
    plt.title('Loss along epochs')
    plt.legend()
    plt.draw()
    plt.savefig(os.path.join(log_dir, img_name + '.png'))
    
    #plt.show()

    
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
    plt.title('Accuracy along epochs')
    plt.legend()
    plt.draw()
    plt.savefig(os.path.join(log_dir, img_name + '.png'))
    
    #plt.show()
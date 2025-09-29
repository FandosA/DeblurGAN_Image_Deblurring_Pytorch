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
    tensor = (tensor + 1) / 2
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
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    
    return tensor

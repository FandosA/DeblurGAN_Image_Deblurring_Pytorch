# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:26:05 2025

@author: fandos
"""

import os
import cv2
import torch
import random
import configargparse
from model import DeblurGAN
from utils import selectDevice, tensorToImage, imageToTensor


if __name__ == "__main__":
    
    # Select parameters for testing
    arg = configargparse.ArgumentParser()
    arg.add_argument('--dataset_path', type=str, default='test_images', help='Dataset path.')
    arg.add_argument('--log_dir', type=str, default='deblurGAN_bs1_lr0.0001_numresblocks9', help='Name of the folder where the files of checkpoints and precision and loss values are stored.')
    arg.add_argument('--checkpoint', type=str, default='checkpoint_14_best_g.pth',help='Checkpoint to use')
    arg.add_argument('--num_resblocks', type=int, default=9, help='Number of residual blocks for the generator.')
    arg.add_argument('--GPU', type=bool, default=True, help='True to train the model in the GPU.')
    args = arg.parse_args()
    
    device = selectDevice(args)
    
    generator = DeblurGAN(n_resblocks=args.num_resblocks)
    state_dict = torch.load(os.path.join(args.log_dir, "checkpoints", args.checkpoint), map_location=device)
    generator.load_state_dict(state_dict)
    generator.to(device)
    generator.eval()
    
    image_paths = []
    
    for root, _, files in os.walk(os.path.join(args.dataset_path, "original")):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(root, file))
    
    with torch.no_grad():
        
        for image_test in image_paths:
            
            image_original = cv2.imread(image_test)
            
            kernel_size = random.choice([3, 5, 7])  # debe ser impar
            sigma = random.uniform(1.0, 7.0)
            image_blurred = cv2.GaussianBlur(image_original, (kernel_size, kernel_size), sigmaX=sigma)
            
            image_blurred_tensor = imageToTensor(image_blurred)
            image_blurred_tensor = torch.unsqueeze(image_blurred_tensor, dim=0).to(device)
            blurred_image_deblurred = generator(image_blurred_tensor)
            blurred_image_deblurred = tensorToImage(blurred_image_deblurred)
            
            image_original_tensor = imageToTensor(image_original)
            image_original_tensor = torch.unsqueeze(image_original_tensor, dim=0).to(device)
            image_original_deblurred = generator(image_original_tensor)
            image_original_deblurred = tensorToImage(image_original_deblurred)
            
            #image_name = image_test.split('\\')[-1]
            
            #cv2.imwrite(os.path.join(args.dataset_path, "upscaled", image_name), image_deblurred)
            cv2.imshow('Original Image', image_original)
            cv2.imshow('Blurred Image', image_blurred)
            cv2.imshow('Original Image Deblurred', image_original_deblurred)
            cv2.imshow('Blurred Image Deblurred', blurred_image_deblurred)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

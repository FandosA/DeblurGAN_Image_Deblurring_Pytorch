# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:18:28 2025

@author: fandos
"""

import os
import cv2
import json
import random
import configargparse
from utils import imageToTensor
from torch.utils.data import Dataset, DataLoader, random_split


class Dataset(Dataset):
    
    def __init__(self, args, device):
        
        self.train_split = args.train_split
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        
        with open(args.dataset_file_path, "r") as file:
            self.images_paths = json.load(file)
        
        
    def __len__(self):
        
        return len(self.images_paths)
    
    
    def __getitem__(self, index):
        
        image_real = cv2.imread(self.images_paths[index])
        
        kernel_size = random.choice([3, 5, 7])
        sigma = random.uniform(1.0, 7.0)
        image_blurred = cv2.GaussianBlur(image_real, (kernel_size, kernel_size), sigmaX=sigma)
        
        image_real = imageToTensor(image_real)
        image_blurred = imageToTensor(image_blurred)
        
        return image_real, image_blurred
    
    
    def loadDataloaders(self):
        
        len_training = int(len(self) * self.train_split)
        len_validation = len(self) - len_training
        
        train_dataset, validate_dataset = random_split(self, [len_training, len_validation])
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        validate_loader = DataLoader(dataset=validate_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        
        print('Training images: ' + str(len(train_dataset)) + '/' + str(len(self)))
        print('Validation images: ' + str(len(validate_dataset)) + '/' + str(len(self)) + '\n')
        
        return train_loader, validate_loader
    
    
if __name__ == "__main__":
    
    arg = configargparse.ArgumentParser()
    arg.add_argument('--dataset_path', type=str, default='../OpenCV contest/dataset', help='Dataset path.')
    args = arg.parse_args()
    
    image_paths = []

    for root, dirs, files in os.walk(args.dataset_path):
        if os.path.basename(root) == "image_left":
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_paths.append(os.path.join(root, file))

    with open("images_paths.json", 'w') as json_file:
        json.dump(image_paths, json_file, indent=4)
        
    print("Number of images:", str(len(image_paths)))
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
from PIL import Image
import pillow_heif
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms

pillow_heif.register_heif_opener()

class Dataset(Dataset):
    
    def __init__(self, args, train=True):
        
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers    
        
        if train:
            self.transformer = transforms.Compose([
                transforms.RandomCrop((256, 256)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            with open(args.json_file_train_path, "r") as file:
                self.images_paths = json.load(file)
        else:
            # self.transformer = transforms.Compose([
                # transforms.CenterCrop(300),
                # transforms.ToTensor(),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # ])
            self.transformer = transforms.ToTensor()
            with open(args.json_file_val_path, "r") as file:
                self.images_paths = json.load(file)
        
        
    def __len__(self):
        
        return len(self.images_paths)
    
    
    def __getitem__(self, index):
        
        image_path = self.images_paths[index]
        
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            
            image_real = cv2.imread(image_path)
            if image_real is None:
                raise RuntimeError(f"Couldn't read image': {image_path}")
            
        elif image_path.lower().endswith(('.heic')):
            
            try:
                pil_img = Image.open(image_path)
                
                if hasattr(pil_img, "n_frames") and pil_img.n_frames > 1:
                    pil_img.seek(0)
                    
                image_real = np.array(pil_img.convert("RGB"))
                image_real = cv2.cvtColor(image_real, cv2.COLOR_RGB2BGR)
                
            except Exception as e:
                raise RuntimeError(f"Error when reading HEIC image {image_path}: {e}")
                
        else:
            
            raise RuntimeError(f"Image format not supported: {image_path}")
                
        kernel_size = random.choice([3, 5, 7])
        sigma = random.uniform(2.0, 9.0)
        
        image_real = cv2.cvtColor(image_real, cv2.COLOR_BGR2RGB)
        image_real = Image.fromarray(image_real)
        image_real = self.transformer(image_real)
        
        blur = transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma))
        image_blurred = blur(image_real)
        
        image_real = image_real * 2 - 1
        image_blurred = image_blurred * 2 - 1
        
        return image_real, image_blurred
    
    
if __name__ == "__main__":
    
    arg = configargparse.ArgumentParser()
    arg.add_argument('--dataset_path', type=str, default=r'../OpenCV contest/dataset;D:\Fotos;D:\Mis Cosas;D:\Osprean', help='Dataset paths.')
    arg.add_argument('--train_split', type=float, default=0.85, help='Percentage of the dataset to use for training.')
    arg.add_argument('--min_width', type=int, default=300, help='Minimum width for the images to include inthe dataset.')
    arg.add_argument('--min_height', type=int, default=300, help='Minimum height for the images to include inthe dataset.')
    args = arg.parse_args()
    
    paths = args.dataset_path.split(";")
    image_paths = []

    for path in paths:
        
        for root, dirs, files in os.walk(path):
            
            if os.path.normpath(path) == os.path.normpath("../OpenCV contest/dataset"):
                if os.path.basename(root) != "image_left":
                    continue

            for file in files:
                
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    
                    full_path = os.path.join(root, file)
                    
                    img = cv2.imread(full_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        if w >= args.min_width and h >= args.min_height:
                            image_paths.append(full_path)
                            
                elif file.lower().endswith(('.heic')):
                    
                    full_path = os.path.join(root, file)
                    
                    try:
                        img_pil = Image.open(full_path)
                        
                        # Si tiene varios frames, quedarse con el primero
                        if hasattr(img_pil, "n_frames") and img_pil.n_frames > 1:
                            img_pil.seek(0)
                                
                        # Convertir a RGB y luego a array NumPy
                        img_rgb = np.array(img_pil.convert("RGB"))
                        
                        h, w = img_rgb.shape[:2]
                        if w >= args.min_width and h >= args.min_height:
                            image_paths.append(full_path)
                            
                    except Exception as e:
                        print(f"Error al convertir {full_path}: {e}")
    
    random.shuffle(image_paths)

    total = len(image_paths)
    train_size = int(total * args.train_split)
    train_paths = image_paths[:train_size]
    val_paths = image_paths[train_size:]

    with open("train_paths_augmented.json", 'w') as f:
        json.dump(train_paths, f, indent=4)

    with open("val_paths_augmented.json", 'w') as f:
        json.dump(val_paths, f, indent=4)

    print(f"Total images: {total}")
    print(f"Train images: {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")

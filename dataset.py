import os
import cv2
import json
import random
import configargparse
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class Dataset(Dataset):
    
    def __init__(self, args, train=True):
        
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers    
        
        if train:
            self.transformer = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ])
            with open(args.json_file_train_path, "r") as file:
                self.images_paths = json.load(file)
        else:
            self.transformer = transforms.ToTensor()
            with open(args.json_file_val_path, "r") as file:
                self.images_paths = json.load(file)
        
        
    def __len__(self):
        
        return len(self.images_paths)
    
    
    def __getitem__(self, index):
        
        image_real = cv2.imread(self.images_paths[index])
        kernel_size = 5
        sigma = 5.0
        
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
    arg.add_argument('--dataset_path', type=str, default=r'dataset;D:\Pictures', help='Dataset paths.')
    arg.add_argument('--train_split', type=float, default=0.9, help='Percentage of the dataset to use for training.')
    args = arg.parse_args()
    
    paths = args.dataset_path.split(";")
    image_paths = []

    for path in paths:
        
        for root, dirs, files in os.walk(path):

            for file in files:
            
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
    
    random.shuffle(image_paths)

    total = len(image_paths)
    train_size = int(total * args.train_split)
    train_paths = image_paths[:train_size]
    val_paths = image_paths[train_size:]

    with open("images_paths_train_ff.json", 'w') as f:
        json.dump(train_paths, f, indent=4)

    with open("images_paths_validation_ff.json", 'w') as f:
        json.dump(val_paths, f, indent=4)

    print(f"Total images: {total}")
    print(f"Train images: {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")

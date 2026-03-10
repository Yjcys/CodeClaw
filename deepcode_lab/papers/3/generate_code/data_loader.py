# Data Loader for SRGAN

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import os

class SRDataset(Dataset):
    def __init__(self, image_dir, upscale_factor=4, transform=None):
        self.image_dir = image_dir
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Convert to numpy array
        img_np = np.array(image)
        
        # Get low-resolution image by applying Gaussian filter and downsampling
        # For simplicity, we'll use bicubic downsampling here
        lr_image = image.resize((image.width // self.upscale_factor, image.height // self.upscale_factor), Image.BICUBIC)
        
        # Convert to tensor
        lr_tensor = transforms.ToTensor()(lr_image)
        hr_tensor = transforms.ToTensor()(image)
        
        return lr_tensor, hr_tensor


def get_data_loader(image_dir, batch_size=16, shuffle=True, num_workers=4):
    transform = transforms.Compose([
        transforms.RandomCrop(96),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    
    dataset = SRDataset(image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader
# src/dataset.py
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image # Import the Pillow library

class OASISDataset(Dataset):
    def __init__(self, data_dir, image_ids, num_classes, transforms=None):
        self.data_dir = data_dir
        self.ids = image_ids
        self.num_classes = num_classes
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # 1. Define file paths
        # Assumes your files are named like 'ID_123.png'
        image_path = os.path.join(self.data_dir, "images", f"{self.ids[index]}.png")
        mask_path = os.path.join(self.data_dir, "masks", f"{self.ids[index]}.png")
        
        # 2. Load the PNG files using Pillow and convert to NumPy arrays
        # .convert("L") ensures the image is loaded as single-channel grayscale
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path)
        image = np.array(image)
        mask = np.array(mask)
        
        # 3. Apply transformations (augmentation) if any
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # 4. Convert to PyTorch Tensors
        # Add a channel dimension for the grayscale image: [H, W] -> [1, H, W]
        image = torch.from_numpy(image).float().unsqueeze(0)
        mask = torch.from_numpy(mask).long()

        # 5. One-hot encode the mask
        mask_one_hot = F.one_hot(mask, num_classes=self.num_classes)
        # Permute from [H, W, C] to the required [C, H, W] format
        mask_one_hot = mask_one_hot.permute(2, 0, 1).float()
        
        return image, mask_one_hot
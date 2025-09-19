# src/dataset.py

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import numpy as np
from PIL import Image

class OASISDataset(Dataset):
    def __init__(self, image_dir, mask_dir, num_classes, transforms=None):
        """
        Args:
            image_dir (str): Directory with all the input images.
            mask_dir (str): Directory with all the segmentation masks.
            num_classes (int): Number of classes for one-hot encoding.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.transforms = transforms
        # Get a sorted list of image filenames from the image directory
        self.images = sorted([f for f in os.listdir(image_dir) if f.startswith('case_')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Get the image filename for the given index
        image_filename = self.images[index]
        image_path = os.path.join(self.image_dir, image_filename)
        
        # Generate the corresponding mask filename by replacing the prefix
        mask_filename = image_filename.replace("case_", "seg_")
        mask_path = os.path.join(self.mask_dir, mask_filename)
        
        # Load image and mask using Pillow
        image = Image.open(image_path).convert("L") # Convert to grayscale
        mask = Image.open(mask_path)
        
        image = np.array(image)
        mask = np.array(mask)
        mask_np = np.array(mask)
        mask_np[mask_np == 85] = 1
        mask_np[mask_np == 170] = 2
        mask_np[mask_np == 255] = 3
        mask = mask_np
        
        # Apply transformations (augmentation) if any
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Convert to PyTorch Tensors
        image = torch.from_numpy(image).float().unsqueeze(0) # [H, W] -> [1, H, W]
        mask = torch.from_numpy(mask).long()

        # One-hot encode the mask
        mask_one_hot = F.one_hot(mask, num_classes=self.num_classes)
        # Permute from [H, W, C] to the required [C, H, W] format
        mask_one_hot = mask_one_hot.permute(2, 0, 1).float()
        
        return image, mask_one_hot
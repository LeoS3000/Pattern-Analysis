# src/dataset.py

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import numpy as np
import nibabel as nib  # Use nibabel for loading Nifti files

class ProstateNiftiDataset(Dataset):
    def __init__(self, image_dir, mask_dir, num_classes, transforms=None):
        """
        Args:
            image_dir (str): Directory with all the 3D Nifti input images. [cite: 210]
            mask_dir (str): Directory with all the 3D Nifti segmentation masks. [cite: 210]
            num_classes (int): Number of classes for one-hot encoding.
            transforms (callable, optional): Optional 3D transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.transforms = transforms
        # Get a sorted list of Nifti filenames (.nii or .nii.gz)
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.nii', '.nii.gz'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # --- Path Construction ---
        image_filename = self.images[index]
        image_path = os.path.join(self.image_dir, image_filename)
        # Mask filenames are assumed to be identical to image filenames
        mask_path = os.path.join(self.mask_dir, image_filename)

        # --- Data Loading (from Appendix B, Listing 2) ---
        # Load the Nifti file using nibabel [cite: 388]
        image_nifti = nib.load(image_path)
        mask_nifti = nib.load(mask_path)

        # Get the image data as a numpy array 
        # Ensure data is float for the model input
        image_data = image_nifti.get_fdata(caching='unchanged').astype(np.float32)
        # Ensure mask data is long/int for one-hot encoding
        mask_data = mask_nifti.get_fdata(caching='unchanged').astype(np.int64)

        # --- Data Transformation ---
        # TODO: Add 3D data augmentation here if needed, e.g., using libraries like TorchIO or Rising
        if self.transforms:
            # Note: 3D transforms will require a different library than Albumentations for 2D
            pass

        # --- Tensor Conversion and Reshaping ---
        # Convert numpy arrays to PyTorch Tensors
        image_tensor = torch.from_numpy(image_data)
        mask_tensor = torch.from_numpy(mask_data)

        # Reshape to fit PyTorch Conv3d format: [C, D, H, W]
        # nibabel loads as [H, W, D], so we permute to [D, H, W]
        image_tensor = image_tensor.permute(2, 0, 1)
        mask_tensor = mask_tensor.permute(2, 0, 1)

        # Add a channel dimension to the image tensor: [D, H, W] -> [1, D, H, W]
        image_tensor = image_tensor.unsqueeze(0)

        # --- One-Hot Encode the Mask (adapted from your original dataset.py) ---
        # [D, H, W] -> [D, H, W, C] -> [C, D, H, W]
        mask_one_hot = F.one_hot(mask_tensor.long(), num_classes=self.num_classes)
        mask_one_hot = mask_one_hot.permute(3, 0, 1, 2).float()

        return image_tensor, mask_one_hot
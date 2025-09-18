# src/utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

def dice_loss(pred, target, smooth=1.):
    """
    Calculates the Dice Loss.
    Args:
        pred (torch.Tensor): The model's raw output logits (B, C, H, W).
        target (torch.Tensor): The one-hot encoded ground truth mask (B, C, H, W).
        smooth (float): A smoothing factor to avoid division by zero.
    Returns:
        torch.Tensor: The calculated Dice loss.
    """
    pred = F.softmax(pred, dim=1)
    
    # Flatten the tensors
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = (pred * target).sum()
    dice_coeff = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice_coeff

def dice_score(pred, target, smooth=1.):
    """
    Calculates the Dice Similarity Coefficient for each class.
    Args:
        pred (torch.Tensor): The model's raw output logits (B, C, H, W).
        target (torch.Tensor): The one-hot encoded ground truth mask (B, C, H, W).
        smooth (float): A smoothing factor.
    Returns:
        list: A list of DSC scores for each class.
    """
    pred = F.softmax(pred, dim=1)
    # Convert prediction to a hard, one-hot encoded mask
    pred_mask = torch.argmax(pred, dim=1)
    pred_one_hot = F.one_hot(pred_mask, num_classes=target.shape[1]).permute(0, 3, 1, 2).float()

    dice_per_class = []
    # Iterate over each class (channel)
    for i in range(target.shape[1]):
        pred_c = pred_one_hot[:, i, :, :].contiguous().view(-1)
        target_c = target[:, i, :, :].contiguous().view(-1)
        
        intersection = (pred_c * target_c).sum()
        score = (2. * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
        dice_per_class.append(score.item())
        
    return dice_per_class


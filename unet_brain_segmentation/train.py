# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm

# Import your custom modules from the 'src' directory
from src.dataset import ProstateNiftiDataset
from src.model import UNet
from src.utils import dice_loss, dice_score, save_checkpoint

import torchio as tio

# Define the 3D augmentation pipeline
transforms = tio.Compose([
    tio.RandomFlip(axes=('LR',)),          # Randomly flip left-right with a 50% chance
    tio.RandomAffine(
        scales=(0.9, 1.2),                 # Randomly scale the image
        degrees=15,                        # Randomly rotate by up to 15 degrees
        isotropic=True,
    ),
    tio.RandomNoise(std=0.01),             # Add a bit of random noise
    tio.RandomBlur(std=(0, 1)),            # Apply a random blur
])

def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    """Runs one full epoch of training."""
    loop = tqdm(loader, leave=True)
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass with Automatic Mixed Precision
        with torch.amp.autocast(device_type='cuda'):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())

def validate_model(loader, model, device, num_classes):
    """Validates the model and returns DSC scores."""
    model.eval()
    all_dsc_scores = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            scores = dice_score(preds, y)
            all_dsc_scores.append(scores)
    
    # Calculate average DSC for each class across all batches
    avg_dsc_per_class = torch.tensor(all_dsc_scores).mean(axis=0).tolist()
    
    model.train() # Set model back to training mode
    return avg_dsc_per_class

def main(args):
    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- Data Loading ---
    train_img_dir = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
    train_mask_dir = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"
    # TODO: You will need to create your own validation split from this data.

    NUM_CLASSES = 6

    # Instantiate the datasets for training and validation
    # Instantiate the new dataset
    train_dataset = ProstateNiftiDataset(
        image_dir=train_img_dir,
        mask_dir=train_mask_dir,
        num_classes=NUM_CLASSES,
        transforms=transforms # <-- Here is where you pass the pipeline
    )

    # Your validation set should typically NOT have augmentations
    val_dataset = ProstateNiftiDataset(
        image_dir=val_img_dir,
        mask_dir=val_mask_dir,
        num_classes=NUM_CLASSES,
        transforms=None # <-- No augmentations for validation
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)

    # Model, Loss, Optimizer 
    model = UNet(n_channels=1, n_classes=args.num_classes).to(device)
    bce_loss = nn.CrossEntropyLoss()
    
    def combined_loss(pred, target):
        return bce_loss(pred, target) + dice_loss(pred, target)


    # Optimizer and LR Scheduler
    initial_lr = 5e-4
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 0.985 ** epoch
    )
    scaler = torch.cuda.amp.GradScaler()

    # Training Loop
    best_dsc = 0.0
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        train_one_epoch(train_loader, model, optimizer, combined_loss, scaler, device)
        
        # Step the learning rate scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Current learning rate: {current_lr:.6f}")
        
        # Validation
        dsc_scores = validate_model(val_loader, model, device, args.num_classes)
        avg_dsc = sum(dsc_scores[1:]) / (args.num_classes - 1) # Avg DSC of foreground classes
        
        print(f"Validation DSC (per class): {[round(s, 4) for s in dsc_scores]}")
        print(f"Average Foreground DSC: {avg_dsc:.4f}")

        # Save model if its the best one so far
        if avg_dsc > best_dsc:
            best_dsc = avg_dsc
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, directory=args.checkpoint_dir, filename="best_model.pth.tar")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train UNet for Brain Segmentation')
    # ... all your parser.add_argument lines ...
    # Example arguments (add your actual arguments as needed):
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    # parser.add_argument('--lr', type=float, default=5e-4)  # Not needed, now hardcoded
    # Add any other arguments you need
    args = parser.parse_args()
    main(args)
# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm

# Import your custom modules from the 'src' directory
from src.dataset import OASISDataset
from src.model import UNet
from src.utils import dice_loss, dice_score, save_checkpoint

def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    """Runs one full epoch of training."""
    loop = tqdm(loader, leave=True)
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass with Automatic Mixed Precision
        with torch.cuda.amp.autocast():
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

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description='Train UNet for Brain Segmentation')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--num-classes', type=int, default=4, help='Number of segmentation classes (including background)')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', help='Path to save models')
    args = parser.parse_args()

    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- Data Loading ---
# Define the paths based on the new structure
train_img_dir = os.path.join(args.data_dir, "keras_png_slices_train")
train_mask_dir = os.path.join(args.data_dir, "keras_png_slices_seg_train")
val_img_dir = os.path.join(args.data_dir, "keras_png_slices_validate")
val_mask_dir = os.path.join(args.data_dir, "keras_png_slices_seg_validate")

# Instantiate the datasets for training and validation
train_dataset = OASISDataset(
    image_dir=train_img_dir, 
    mask_dir=train_mask_dir, 
    num_classes=args.num_classes
)
val_dataset = OASISDataset(
    image_dir=val_img_dir, 
    mask_dir=val_mask_dir, 
    num_classes=args.num_classes
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)

    # --- Model, Loss, Optimizer ---
    model = UNet(n_channels=1, n_classes=args.num_classes).to(device)
    bce_loss = nn.CrossEntropyLoss()
    
    def combined_loss(pred, target):
        return bce_loss(pred, target) + dice_loss(pred, target)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    
    # --- Training Loop ---
    best_dsc = 0.0
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        train_one_epoch(train_loader, model, optimizer, combined_loss, scaler, device)
        
        # Validation
        dsc_scores = validate_model(val_loader, model, device, args.num_classes)
        avg_dsc = sum(dsc_scores[1:]) / (args.num_classes - 1) # Avg DSC of foreground classes
        
        print(f"Validation DSC (per class): {[round(s, 4) for s in dsc_scores]}")
        print(f"Average Foreground DSC: {avg_dsc:.4f}")

        # Save model if it's the best one so far
        if avg_dsc > best_dsc:
            best_dsc = avg_dsc
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, directory=args.checkpoint_dir, filename="best_model.pth.tar")

if __name__ == "__main__":
    main()
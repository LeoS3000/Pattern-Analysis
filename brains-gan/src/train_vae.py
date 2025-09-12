# --------------------------------------------------------------
# train_vae.py
# --------------------------------------------------------------
import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as T

# ------------------------------------------------------------------
# 1️⃣  Imports – adjust these to match your project layout
# ------------------------------------------------------------------
from dataset import PNGDataset          # <-- your dataset class
from model_vae import VAE               # <-- either VAEGAN or a thin wrapper called VAE

# ------------------------------------------------------------------
# 2️⃣  Basic setup
# ------------------------------------------------------------------
print("=== VAE training start ===")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ------------------------------------------------------------------
# 3️⃣  Hyper‑parameters
# ------------------------------------------------------------------
latent_dim   = 128
batch_size   = 16
epochs       = 30          # feel free to increase
lr           = 1e-3
img_dir      = '../data/keras_png_slices_train'
out_dir      = 'out/vae'

# ------------------------------------------------------------------
# 4️⃣  Make sure the output folder exists
# ------------------------------------------------------------------
os.makedirs(out_dir, exist_ok=True)

# ------------------------------------------------------------------
# 5️⃣  Dataset & DataLoader
# ------------------------------------------------------------------
# If your PNGDataset does NOT already normalise, wrap it with a transform:
#   transform = T.Compose([T.ToTensor()])   # converts to [0,1] and (C,H,W)
#   dataset   = PNGDataset(img_dir, transform=transform)
# Otherwise just instantiate it as you already do.
dataset = PNGDataset(img_dir)
loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# ------------------------------------------------------------------
# 6️⃣  Model, optimiser, loss
# ------------------------------------------------------------------
model = VAE(latent_dim).to(device)          # <-- VAEGAN wrapper works as VAE
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mse_loss = torch.nn.MSELoss()

def vae_loss(recon, x, mu, logvar):
    """Standard VAE loss = reconstruction (MSE) + KL divergence."""
    recon_loss = mse_loss(recon, x)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# ------------------------------------------------------------------
# 7️⃣  Training loop
# ------------------------------------------------------------------
for epoch in range(epochs):
    model.train()                     # make sure we are in training mode
    for i, x in enumerate(loader):
        x = x.to(device)              # (B, C, H, W) – already normalised

        optimizer.zero_grad()
        recon, mu, logvar = model(x)  # forward
        loss = vae_loss(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()

        # --------------------------------------------------------------
        # Logging & checkpointing
        # --------------------------------------------------------------
        if i % 100 == 0:
            print(f"[Epoch {epoch:02d} | Step {i:04d}] loss = {loss.item():.4f}")

            # Save a *detached* grid of reconstructions (no grads attached)
            model.eval()
            with torch.no_grad():
                save_image(
                    recon[:8].cpu().detach(),
                    os.path.join(out_dir, f"recon_e{epoch}_s{i}.png")
                )
            model.train()

    # optional: save a checkpoint at the end of each epoch
    ckpt_path = os.path.join(out_dir, f"ckpt_epoch{epoch}.pth")
    torch.save(model.state_dict(), ckpt_path)

print("=== Training finished ===")
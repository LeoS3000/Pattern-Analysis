# models/discriminator.py
import torch
from torch import nn

class PatchDiscriminator(nn.Module):
    """
    A lightweight PatchGAN discriminator.
    It classifies each N×N patch as real/fake and averages the result.
    Works well for 256×256 images and is cheap to train.
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            # Conv => LeakyReLU (no BN on first layer, per PatchGAN paper)
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # (base) → (base*2)
            nn.Conv2d(base_channels, base_channels*2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base_channels*2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # (base*2) → (base*4)
            nn.Conv2d(base_channels*2, base_channels*4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base_channels*4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # (base*4) → (base*8)
            nn.Conv2d(base_channels*4, base_channels*8, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(base_channels*8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Final 1‑channel output (real/fake score per patch)
            nn.Conv2d(base_channels*8, 1, 4, 1, 1)   # output shape: (B,1,H/16,W/16)
        )

    def forward(self, x):
        return self.net(x)   # raw logits – we will apply BCEWithLogitsLoss later
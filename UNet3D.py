import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv3d → BN → ReLU) twice."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.double_conv(x)

class GenDown(nn.Module):
    """Downscaling with MaxPool then DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x):
        return self.pool_conv(x)

class GenUp(nn.Module):
    """Upscaling then DoubleConv (with skip connection)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # ConvTranspose3d halves channel dim and doubles spatial dims
        self.up = nn.ConvTranspose3d(in_ch, in_ch//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if needed (handles odd-sized inputs)
        diffZ = skip.size(2) - x.size(2)
        diffY = skip.size(3) - x.size(3)
        diffX = skip.size(4) - x.size(4)
        x = F.pad(x, [diffX//2, diffX-diffX//2,
                      diffY//2, diffY-diffY//2,
                      diffZ//2, diffZ-diffZ//2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=6, out_classes=3, features=[32, 64, 128, 256]):
        super().__init__()

        self.noise_proj = nn.Sequential(
            nn.ConvTranspose3d(noise_dim, 3*2, kernel_size=4, stride=4),      # 1→4
            nn.BatchNorm3d(3*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(3*2, 3, kernel_size=33, stride=33),# 4→136
            nn.BatchNorm3d(3),
            nn.ReLU(inplace=True),
        )

        # Encoder
        self.inc   = DoubleConv(in_channels, features[0])
        self.down1 = GenDown(features[0], features[1])
        self.down2 = GenDown(features[1], features[2])
        self.down3 = GenDown(features[2], features[3])
        # Decoder
        self.up2   = GenUp(features[3], features[2])
        self.up1   = GenUp(features[2], features[1])
        self.up0   = GenUp(features[1], features[0])
        # Final 1×1×1 conv to map to desired classes
        self.outc  = nn.Conv3d(features[0], out_classes, kernel_size=1)

    def forward(self, x, z):

        B, _, D, H, W = x.shape

        z_vol = z.view(B, noise_dim, 1, 1, 1)
        z_flat = self.noise_proj(z_vol)              # [B, project_to_ch * 64 * 64 * 64]
        z_final = z_flat.view(B, 3, D, H, W)
        
        x_with_noise = torch.cat([x, z_final], dim=1)

        x0 = self.inc(x_with_noise)        # → [B, f0, D, H, W]
        x1 = self.down1(x0)     # → [B, f1, D/2, H/2, W/2]
        x2 = self.down2(x1)     # → [B, f2, D/4, H/4, W/4]
        x3 = self.down3(x2)     # → [B, f3, D/8, H/8, W/8]
        x  = self.up2(x3, x2)   # → [B, f2, D/4, H/4, W/4]
        x  = self.up1(x,  x1)   # → [B, f1, D/2, H/2, W/2]
        x  = self.up0(x,  x0)   # → [B, f0, D,   H,   W  ]
        return self.outc(x)     # → [B, out_classes, D, H, W]



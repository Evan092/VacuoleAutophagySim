import torch
import torch.nn as nn
from Constants import noise_dim
import torch.nn.functional as F



import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # project to lower‐dim Q/K/V
        self.query_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv   = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels,      kernel_size=1)
        self.gamma      = nn.Parameter(torch.zeros(1))  # learnable residual scale

    def forward(self, x):
        B, C, D, H, W = x.shape
        # compute Q, K, V
        q = self.query_conv(x).view(B, -1, D*H*W)      # B × C’ × N
        k = self.key_conv(x).view(B, -1, D*H*W)        # B × C’ × N
        v = self.value_conv(x).view(B,  C, D*H*W)      # B × C  × N

        # attention map (B × N × N)
        attn = torch.softmax(q.permute(0,2,1) @ k, dim=-1)

        # aggregate V
        out = (v @ attn).view(B, C, D, H, W)           # B × C × D × H × W

        return self.gamma * out + x




class DoubleConv(nn.Module):
    """(Conv3d → BN → ReLU) twice."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm3d(out_ch),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.ReLU(inplace=False),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            #nn.BatchNorm3d(out_ch),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.ReLU(inplace=False),
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


class ProbHead(nn.Module):
    def __init__(self, features, out_classes):
        super().__init__()
        self.prob_head = nn.Sequential(
            nn.Conv3d(features[0], features[1], 1, bias=False),             # 16 → 32 (cheap)
            nn.GroupNorm(4, features[1]), nn.LeakyReLU(inplace=True),
            nn.Conv3d(features[1], features[2], 3, padding=1, bias=False),  # 32 → 64
            nn.GroupNorm(8, features[2]), nn.LeakyReLU(inplace=True),
            nn.Dropout3d(0.1),
            nn.Conv3d(features[2], out_classes, 1, bias=True)
        )             # your conv/gn/relu/... stack

    def forward(self, x):
        # upstream is in mixed precision; force FP32 math here
        with torch.amp.autocast(device_type="cuda", enabled=False):
            return self.prob_head(x.float()) # cast activations to FP32 for this block
        

class DirHead(nn.Module):
    def __init__(self, features, out_classes):
        super().__init__()
        self.outc  = nn.Conv3d(features, out_classes, kernel_size=1)        # your conv/gn/relu/... stack

    def forward(self, x):
        # upstream is in mixed precision; force FP32 math here
        with torch.amp.autocast(device_type="cuda", enabled=False):
            return self.outc(x.float()) # cast activations to FP32 for this block

class UNet3D(nn.Module):
    def __init__(self, in_channels=9, out_classes=3, features=[16, 32, 64, 128, 256]):
        super().__init__()

        self.noise_proj = nn.Sequential(
            # 1→4
            nn.ConvTranspose3d(noise_dim, 8, kernel_size=4, stride=4),
            nn.GroupNorm(num_groups=8, num_channels=8),
            nn.ReLU(inplace=False),

            # 4→200  (3*50 + 50 = 200)
            nn.ConvTranspose3d(8, 2, kernel_size=50, stride=50),
            nn.GroupNorm(num_groups=2, num_channels=2),
            nn.ReLU(inplace=False),
        )

        # Encoder
        self.inc   = DoubleConv(in_channels, features[0])
        self.down1 = GenDown(features[0], features[1])
        self.down2 = GenDown(features[1], features[2])
        self.down3 = GenDown(features[2], features[3])
        self.down4 = GenDown(features[3], features[4])

        #Attention
        self.attention1 = SelfAttention3D(features[4])
        self.attention2 = SelfAttention3D(features[4])
        self.attention3 = SelfAttention3D(features[4])
        # Decoder
        self.up3   = GenUp(features[4], features[3])
        self.up2   = GenUp(features[3], features[2])
        self.up1   = GenUp(features[2], features[1])
        self.up0   = GenUp(features[1], features[0])
        # Final 1×1×1 conv to map to desired classes
        self.outc  = DirHead(features[0], out_classes)


        self.convD3 = nn.Conv3d(features[3], features[3], kernel_size=3, padding=1, stride=1)
        self.convD4 = nn.Conv3d(features[4], features[4], kernel_size=3, padding=1, stride=1)

        self.convU3 = nn.Conv3d(features[3], features[3], kernel_size=3, padding=1, stride=1)
        self.convU2 = nn.Conv3d(features[2], features[2], kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU(inplace=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.prob_head = ProbHead(features, out_classes)

    def forward(self, x, steps):

        #x = a[:, :3, ...] #vectors
        #y = a[:, 3:, ...] #probabilities

        B, _, D, H, W = x.shape

        z_final = torch.empty(B, 2, D, H, W, device=x.device, dtype=x.dtype).normal_(0.0, 0.1)


        step_channel = torch.full((B, 1, D, H, W), float(steps), device=x.device, dtype=x.dtype)

        x_with_noise = torch.cat([x, z_final, step_channel], dim=1)
        #y_with_noise = torch.cat([y, z_final, step_channel], dim=1)
        

        x0 = self.inc(x_with_noise)        # → [B, f0, D, H, W]
        x1 = self.down1(x0)     # → [B, f1, D/2, H/2, W/2]
        x2 = self.down2(x1)     # → [B, f2, D/4, H/4, W/4]
        x3 = self.down3(x2)     # → [B, f3, D/8, H/8, W/8]
        x3 = self.convD3(x3)
        x3 = self.relu(x3)
        x4 = self.down4(x3)     # → [B, f3, D/8, H/8, W/8]
        x4 = self.convD4(x4)
        x4 = self.relu(x4)

        x4 = self.attention1(x4)
        x4 = self.attention2(x4)
        x4 = self.attention3(x4)

        x  = self.up3(x4, x3)   # → [B, f2, D/4, H/4, W/4]
        x = self.convU3(x)
        x = self.relu(x)

        x  = self.up2(x, x2)   # → [B, f2, D/4, H/4, W/4]

        x = self.convU2(x)
        x = self.relu(x)

        x  = self.up1(x,  x1)   # → [B, f1, D/2, H/2, W/2]
        x  = self.up0(x,  x0)   # → [B, f0, D,   H,   W  ]

        #result = self.outc(x)

        return torch.cat([self.tanh(self.outc(x)), self.prob_head(x)], dim=1)



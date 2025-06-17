import torch.nn as nn

class Discriminator3D(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super(Discriminator3D, self).__init__()
        # Downsample block: Conv3d -> BatchNorm3d -> LeakyReLU
        def down_block(in_ch, out_ch, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=stride, padding=padding, bias=False),
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )
        
        # Initial conv (no batchnorm)
        self.initial = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.down1 = down_block(base_channels, base_channels * 2)   # → 2× down
        self.down2 = down_block(base_channels * 2, base_channels * 4) # → 4× down
        self.down3 = down_block(base_channels * 4, base_channels * 8) # → 8× down
        # You can add more downsampling layers if your D, H, W are large.
        
        # Final layer: reduce to 1 channel
        self.final = nn.Conv3d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1, bias=False)
        # Output shape: [B,1,D',H',W'] (where D'/H'/W' depend on input size and downsampling)
    
    def forward(self, x):
        """
        x: [B, 3, D, H, W] tensor of voxel scores (raw logits or softmaxed)
        returns: [B, 1, D', H', W'] raw logits for “real vs. fake” on each patch/voxel region
        """
        x = self.initial(x)   # → [B, base, D/2, H/2, W/2]
        x = self.down1(x)     # → [B, base*2, D/4, H/4, W/4]
        x = self.down2(x)     # → [B, base*4, D/8, H/8, W/8]
        x = self.down3(x)     # → [B, base*8, D/16, H/16, W/16]
        out = self.final(x)   # → [B, 1, D', H', W']
        return out



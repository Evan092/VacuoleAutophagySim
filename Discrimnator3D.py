import torch.nn as nn
from torch.nn.utils import spectral_norm

from ScheduleDropout import ScheduledDropout

class Discriminator3D(nn.Module):
    def __init__(self, in_channels=6, base_channels=8):
        super(Discriminator3D, self).__init__()
        # Downsample block: Conv3d -> BatchNorm3d -> LeakyReLU -> Dropout
        def down_block(in_ch, out_ch, stride=2, padding=1):
            return nn.Sequential(
                spectral_norm(nn.Conv3d(in_ch, out_ch, kernel_size=4, stride=stride, padding=padding, bias=False)),
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                ScheduledDropout(p_start=0.6, p_end=0.0, T_max=900)
                #nn.Dropout(0.45)
            )
        
        # Initial conv (no batchnorm)
        self.initial = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            ScheduledDropout(p_start=0.6, p_end=0.0, T_max=900)
            #nn.Dropout(0.45)
        )
        
        self.down1 = down_block(base_channels, base_channels * 2)   # → 2× down
        self.down2 = down_block(base_channels * 2, base_channels * 4) # → 4× down
        self.down3 = down_block(base_channels * 4, base_channels * 8) # → 8× down
        self.down4 = down_block(base_channels * 8, base_channels * 8) # → 8× down
        self.down5 = down_block(base_channels * 8, base_channels * 16) # → 8× down
        
        # Final layer: reduce to 1 channel
        self.final = spectral_norm(
            nn.Conv3d(base_channels * 16, 1, kernel_size=4, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.final(x)
        return x




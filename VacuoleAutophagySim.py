import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F



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


# ----------------------------------------
# 1) Data parsing and Dataset
# ----------------------------------------

def parse_VTK(path):
    """
    Read a text-based voxel file where each line specifies a solid block:
      ID Type x1 x2 y1 y2 z1 z2
    Build and return a 3D numpy volume (depth, height, width) with 1s for occupied voxels.
    """
    vol_size    = (128, 128, 128) #default size

    data = []

    x,y,z = -1,0,0

    with open(path, 'r') as vtk:
        for line in vtk:
            line = str(line).strip()
            if (line.lower().startswith("dimensions")): #dimmensions are in the .vtk, copy them.
                dimms = line.split(" ")
                dimms[1] = int(dimms[1])
                dimms[2] = int(dimms[2])
                dimms[3] = int(dimms[3])
                vol_size = (dimms[1], dimms[2], dimms[3])
                vol = np.zeros((dimms[1], dimms[2], dimms[3]), dtype=np.float32)
            if (line and line[0].isdigit()):
                data.extend(int(v) for v in line.split())

                

    arr = np.array(data, dtype=np.int32)
    arr = arr.reshape((vol_size[3], vol_size[2], vol_size[1])) #VTK is "Fill X Axis fastest", Numpy makes last param fastest, so we need z,y,x rather than x,y,z
    volume = arr.transpose(2, 1, 0) #now we flip it to desired x,y,z
    return volume


def parse_voxel_file_labeled(path):
    voxels = []
    max_x = max_y = max_z = 0

    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            id_int = int(parts[0])          # keep the ID
            x1, x2, y1, y2, z1, z2 = map(int, parts[2:])
            voxels.append((id_int, x1, x2, y1, y2, z1, z2))
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)
            max_z = max(max_z, z2)

    volume = np.zeros((max_z + 1, max_y + 1, max_x + 1), dtype=np.float32)
    for id_int, x1, x2, y1, y2, z1, z2 in voxels:
        volume[z1 : z2 + 1, y1 : y2 + 1, x1 : x2 + 1] = id_int

    return volume



def parse_voxel_file(path):
    """
    Read a text-based voxel file where each line specifies a solid block:
      ID Type x1 x2 y1 y2 z1 z2
    Build and return a 3D numpy volume (depth, height, width) with 1s for occupied voxels.
    """
    voxels = []
    max_x = max_y = max_z = 0

    # Parse each line and track the maximum extents
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            _, cell_type, x1, x2, y1, y2, z1, z2 = parts
            x1, x2 = int(x1), int(x2)
            y1, y2 = int(y1), int(y2)
            z1, z2 = int(z1), int(z2)
            voxels.append((cell_type, x1, x2, y1, y2, z1, z2))
            # update volume bounds
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)
            max_z = max(max_z, z2)

    # Initialize empty volume and fill in occupied regions
    vol = np.zeros((3, max_z + 1, max_y + 1, max_x + 1), dtype=np.float32)
    for cell_type, x1, x2, y1, y2, z1, z2 in voxels:
        if cell_type == "Body":
            vol[1, z1 : z2 + 1, y1 : y2 + 1, x1 : x2 + 1] = 1.0
        elif cell_type == "Wall":
            vol[2, z1 : z2 + 1, y1 : y2 + 1, x1 : x2 + 1] = 1.0

    mask_occupied = (vol[1] + vol[2]) == 0  # True wherever neither Body nor Wall was written
    vol[0, mask_occupied] = 1.0

    return vol

class VoxelDataset(Dataset):
    """
    PyTorch Dataset for loading 3D voxel volumes.  
    Returns (input, target) pairs where target == input for autoencoder training.
    """
    def __init__(self, folder, transform=None):
        # collect all .txt or .piff files in folder
        self.paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith('.txt') or f.endswith('.piff')
        ]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if idx >= len(self.paths)-1:
            idx -= 1
        # parse to numpy volume, convert to tensor with channel dim
        vol = parse_voxel_file(self.paths[idx])
        tensor = torch.from_numpy(vol) #.unsqueeze(0)  # shape [1, D, H, W]
        if self.transform:
            tensor = self.transform(tensor)
        vol2 = parse_voxel_file(self.paths[idx+1])
        tensor2 = torch.from_numpy(vol2)#.unsqueeze(0)  # shape [1, D, H, W]
        if self.transform:
            tensor2 = self.transform(tensor2)
        return tensor, tensor2

# ----------------------------------------
# 2) 3D U-Net Components
# ----------------------------------------

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
    def __init__(self, in_channels=3, out_classes=3, features=[32, 64, 128, 256]):
        super().__init__()
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

    def forward(self, x):
        x0 = self.inc(x)        # → [B, f0, D, H, W]
        x1 = self.down1(x0)     # → [B, f1, D/2, H/2, W/2]
        x2 = self.down2(x1)     # → [B, f2, D/4, H/4, W/4]
        x3 = self.down3(x2)     # → [B, f3, D/8, H/8, W/8]
        x  = self.up2(x3, x2)   # → [B, f2, D/4, H/4, W/4]
        x  = self.up1(x,  x1)   # → [B, f1, D/2, H/2, W/2]
        x  = self.up0(x,  x0)   # → [B, f0, D,   H,   W  ]
        return self.outc(x)     # → [B, out_classes, D, H, W]

# ----------------------------------------
# 4) Main Script
# ----------------------------------------

def train(gen_model, disc_model, dataloader, gen_optimizer, disc_optimizer, gen_criterion, disc_criterion, device):
    gen_model.train()
    disc_model.train()
    running_loss = 0.0
    total = 0
    for volumes, targets in dataloader:
        volumes = volumes.to(device)
        targets = targets.to(device)


        #Generate our predicted values
        gen_optimizer.zero_grad()
        gen_outputs = gen_model(volumes)

        gen_loss = gen_criterion(gen_outputs, targets)
    
        disc_optimizer.zero_grad()
        disc_outputs = disc_model(targets)

        real_loss = disc_criterion(disc_outputs, torch.ones_like(disc_outputs))


        fake_output = torch.zeros_like(gen_outputs).scatter_(
            dim=1,
            index=gen_outputs.argmax(dim=1, keepdim=True),
            value=1.0)


        fake_output = fake_output.detach()

        disc_outputs = disc_model(fake_output)

        fake_loss = disc_criterion(disc_outputs, torch.zeros_like(disc_outputs))

        (fake_loss + real_loss).backward()
        disc_optimizer.step()

        fake_output = torch.zeros_like(gen_outputs).scatter_(
            dim=1,
            index=gen_outputs.argmax(dim=1, keepdim=True),
            value=1.0)

        disc_outputs2 = disc_model(fake_output)

        adv_loss = disc_criterion(disc_outputs2, torch.ones_like(disc_outputs2))

        #finalize generated loss
        (gen_loss + adv_loss).backward()
        gen_optimizer.step()

        print("Gen_Loss: ", gen_loss.item())
        print("real_Loss: ", real_loss.item())
        print("fake_Loss: ", fake_loss.item())
        print("adv_Loss: ", adv_loss.item())
        

        running_loss += (gen_loss.item() + real_loss.item() + fake_loss.item() + adv_loss.item()) * volumes.size(0)
        total += volumes.size(0)
    print("Current:", running_loss/total)

    return running_loss / total


def main():
     # User-defined parameters (adjust to your data)
    data_folder = "C:\\Users\\evans\\Documents\\Independent Study\\outputs"
    latent_dim  = 256               # size of VAE bottleneck
    base_ch     = 32                # number of filters in first layer
    batch_size  = 2                 # samples per GPU batch
    epochs      = 100               # training duration
    lr          = 1e-4              # learning rate






    # Setup PyTorch device and data loader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = VoxelDataset(data_folder)

    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    # Instantiate model and begin training
    gen_model = UNet3D()
    disc_model = Discriminator3D()


    gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=lr)
    disc_optimizer = torch.optim.Adam(disc_model.parameters(), lr=lr)
    gen_criterion = nn.CrossEntropyLoss()
    disc_criterion = nn.BCEWithLogitsLoss()

    num_epochs = 20

    for epoch in range(num_epochs):
        train(gen_model, disc_model, loader, gen_optimizer, disc_optimizer, gen_criterion, disc_criterion, device)

    # Save weights for later use
    torch.save(gen_model.state_dict(), "unet3d_vae_checkpoint.pth")
    torch.save(disc_model.state_dict(), "cnn3d_vae_checkpoint.pth")
    print("Checkpoint saved.")


if __name__ == "__main__":
   main()
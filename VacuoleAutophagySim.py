import argparse
import datetime
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
import pandas as pd
from scipy.ndimage    import distance_transform_edt, label
from skimage.feature  import peak_local_max
from skimage.segmentation import watershed
import numpy as np
import datetime


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

def parse_voxel_file_for_ID_matching(path):
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
            cell_ID, cell_type, x1, x2, y1, y2, z1, z2 = parts
            x1, x2 = int(x1), int(x2)
            y1, y2 = int(y1), int(y2)
            z1, z2 = int(z1), int(z2)
            voxels.append((cell_ID, cell_type, x1, x2, y1, y2, z1, z2))
            # update volume bounds
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)
            max_z = max(max_z, z2)

    # Initialize empty volume and fill in occupied regions
    vol = np.zeros((3, max_z + 1, max_y + 1, max_x + 1), dtype=np.float32)
    for cell_ID, cell_type, x1, x2, y1, y2, z1, z2 in voxels:
        if cell_type == "Body":
            vol[1, z1 : z2 + 1, y1 : y2 + 1, x1 : x2 + 1] = cell_ID
        elif cell_type == "Wall":
            vol[2, z1 : z2 + 1, y1 : y2 + 1, x1 : x2 + 1] = cell_ID

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
        inputVol = parse_voxel_file(self.paths[idx])
        inputTensor = torch.from_numpy(inputVol) #.unsqueeze(0)  # shape [1, D, H, W]
        if self.transform:
            inputTensor = self.transform(inputTensor)

        IDVol = parse_voxel_file(self.paths[idx])
        IDTensor = torch.from_numpy(IDVol) #.unsqueeze(0)  # shape [1, D, H, W]

        gt_instances = {}
        # channels: 0=medium (ignore), 1=body, 2=wall
        for ch in [1, 2]:
            # grab all the IDs in this channel (background is encoded as 0)
            ids = np.unique(IDTensor[ch])
            ids = ids[ids != 0]   # drop the 0 background
            for id_ in ids:
                # make a boolean mask for that specific cell (or wall)
                mask = (IDTensor[ch] == id_)
                gt_instances[(ch, id_)] = mask


        outputVol = parse_voxel_file(self.paths[idx+1])
        targetTensor = torch.from_numpy(outputVol)#.unsqueeze(0)  # shape [1, D, H, W]
        if self.transform:
            targetTensor = self.transform(targetTensor)
        return inputTensor, targetTensor, gt_instances


class InferenceDataset(Dataset):
    """
    PyTorch Dataset for loading 3D voxel volumes.  
    Returns (input, target) pairs where target == input for autoencoder training.
    """
    def __init__(self, piff, transform=None):
        # collect all .txt or .piff files in folder
        self.paths = [piff]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if idx >= len(self.paths)-1:
            idx -= 1
        # parse to numpy volume, convert to tensor with channel dim
        inputVol = parse_voxel_file(self.paths[idx])
        inputTensor = torch.from_numpy(inputVol) #.unsqueeze(0)  # shape [1, D, H, W]
        if self.transform:
            inputTensor = self.transform(inputTensor)

        IDVol = parse_voxel_file(self.paths[idx])
        IDTensor = torch.from_numpy(IDVol) #.unsqueeze(0)  # shape [1, D, H, W]

        gt_instances = {}
        # channels: 0=medium (ignore), 1=body, 2=wall
        for ch in [1, 2]:
            # grab all the IDs in this channel (background is encoded as 0)
            ids = np.unique(IDTensor[ch])
            ids = ids[ids != 0]   # drop the 0 background
            for id_ in ids:
                # make a boolean mask for that specific cell (or wall)
                mask = (IDTensor[ch] == id_)
                gt_instances[(ch, id_)] = mask


        outputVol = parse_voxel_file(self.paths[idx+1])
        targetTensor = torch.from_numpy(outputVol)#.unsqueeze(0)  # shape [1, D, H, W]
        if self.transform:
            targetTensor = self.transform(targetTensor)
        return inputTensor, targetTensor, gt_instances

# ----------------------------------------
# 2) 3D U-Net Components
# ----------------------------------------

noise_dim = 32

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

# ----------------------------------------
# 4) Main Script
# ----------------------------------------

def train(gen_model, disc_model, dataloader, gen_optimizer, disc_optimizer, gen_criterion, disc_criterion, device):
    gen_model.train()
    disc_model.train()
    running_loss = 0.0
    running_gen_loss = 0.0
    running_real_loss = 0.0
    running_fake_loss = 0.0
    running_adv_loss = 0.0
    total = 0
    for volumes, targets, _ in dataloader:
        volumes = volumes.to(device)
        targets = targets.to(device)

        B = volumes.shape[0]
        z = torch.randn(B, noise_dim, device=device) * 0.1


        #Generate our predicted values
        gen_optimizer.zero_grad()
        gen_outputs = gen_model(volumes, z)

        gen_loss = gen_criterion(gen_outputs, targets)
    
        disc_optimizer.zero_grad()
        disc_outputs = disc_model(targets)

        real_loss = disc_criterion(disc_outputs, torch.full_like(disc_outputs, 0.9))


        fake_output = torch.zeros_like(gen_outputs).scatter_(
            dim=1,
            index=gen_outputs.argmax(dim=1, keepdim=True),
            value=1.0)


        fake_output = fake_output.detach()

        disc_outputs = disc_model(fake_output)

        fake_loss = disc_criterion(disc_outputs, torch.full_like(disc_outputs, 0.1))

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
        running_gen_loss += (gen_loss.item()) * volumes.size(0)
        running_real_loss += (real_loss.item()) * volumes.size(0)
        running_fake_loss += (fake_loss.item()) * volumes.size(0)
        running_adv_loss += (adv_loss.item()) * volumes.size(0)
        total += volumes.size(0)
    print("Current_gen:", running_gen_loss/total)
    print("Current_real:", running_real_loss/total)
    print("Current_fake:", running_fake_loss/total)
    print("Current_adv:", running_adv_loss/total)
    print("Current:", running_loss/total)

    with open("log.txt", "a") as f:
        f.write("Current_gen:" + str( running_gen_loss/total) + "\n")
        f.write("Current_real:" + str(  running_real_loss/total) + "\n")
        f.write("Current_fake:" + str(  running_fake_loss/total) + "\n")
        f.write("Current_adv:" + str(  running_adv_loss/total) + "\n")
        f.write("Current:" + str(  running_loss/total) + "\n")
        f.write("------------------------------------------" + "\n")

    return running_loss / total



def evaluate(gen_model, disc_model, dataloader, gen_optimizer, disc_optimizer, gen_criterion, disc_criterion, device):
    gen_model.eval()
    running_loss = 0.0
    running_gen_loss = 0.0
    running_real_loss = 0.0
    running_fake_loss = 0.0
    running_adv_loss = 0.0
    total = 0
    for volumes, targets, gt_instances in dataloader:
        volumes = volumes.to(device)
        targets = targets.to(device)

        with torch.no_grad():

            B = volumes.shape[0]
            z = torch.randn(B, noise_dim, device=device) * 0.1


            #Generate our predicted values

            gen_outputs = gen_model(volumes, z)

            gen_loss = gen_criterion(gen_outputs, targets)
        

            disc_outputs = disc_model(targets)

            real_loss = disc_criterion(disc_outputs, torch.full_like(disc_outputs, 0.9))


            fake_output = torch.zeros_like(gen_outputs).scatter_(
                dim=1,
                index=gen_outputs.argmax(dim=1, keepdim=True),
                value=1.0)


            fake_output = fake_output.detach()

            disc_outputs = disc_model(fake_output)

            fake_loss = disc_criterion(disc_outputs, torch.full_like(disc_outputs, 0.1))


            fake_output = torch.zeros_like(gen_outputs).scatter_(
                dim=1,
                index=gen_outputs.argmax(dim=1, keepdim=True),
                value=1.0)

            disc_outputs2 = disc_model(fake_output)

            adv_loss = disc_criterion(disc_outputs2, torch.ones_like(disc_outputs2))

            print(time.now())
            #Post-Process
            pred_label = gen_outputs.detach().cpu().numpy().argmax(axis=0)  # [Z,Y,X]

            all_matches = []
            for ch in [1, 2]:
                # 2) Extract predicted instances for this channel
                pred_instances = {}
                binary, comp_map = (pred_label == ch), None
                comp_map, num = label(binary)
                for inst_id in range(1, num+1):
                    pred_instances[(ch, inst_id)] = (comp_map == inst_id)

                # 3) Gather GT instance keys for this same channel
                gt_keys   = [k for k in gt_instances if k[0] == ch]
                pred_keys = [k for k in pred_instances if k[0] == ch]

                # 4) Build the cost matrix
                cost = np.zeros((len(gt_keys), len(pred_keys)), dtype=np.float32)
                for i, gk in enumerate(gt_keys):
                    gmask = gt_instances[gk]
                    for j, pk in enumerate(pred_keys):
                        pmask = pred_instances[pk]
                        inter = np.logical_and(gmask, pmask).sum()
                        union = np.logical_or(gmask, pmask).sum()
                        iou   = inter / (union + 1e-9)
                        cost[i, j] = -iou

                # 5) Run Hungarian and collect matches
                row_idx, col_idx = linear_sum_assignment(cost)
                for r, c in zip(row_idx, col_idx):
                    gt_key   = gt_keys[r]
                    pred_key = pred_keys[c]
                    all_matches.append((gt_key, pred_key, -cost[r, c]))

                print(time.now())





        print("Gen_Loss: ", gen_loss.item())
        print("real_Loss: ", real_loss.item())
        print("fake_Loss: ", fake_loss.item())
        print("adv_Loss: ", adv_loss.item())
        

        running_loss += (gen_loss.item() + real_loss.item() + fake_loss.item() + adv_loss.item()) * volumes.size(0)
        running_gen_loss += (gen_loss.item()) * volumes.size(0)
        running_real_loss += (real_loss.item()) * volumes.size(0)
        running_fake_loss += (fake_loss.item()) * volumes.size(0)
        running_adv_loss += (adv_loss.item()) * volumes.size(0)
        total += volumes.size(0)
    print("Current_gen:", running_gen_loss/total)
    print("Current_real:", running_real_loss/total)
    print("Current_fake:", running_fake_loss/total)
    print("Current_adv:", running_adv_loss/total)
    print("Current:", running_loss/total)

    return running_loss / total


def runinference(gen_model, volumes, gt_instances, device):
    gen_model.eval()
    running_loss = 0.0
    running_gen_loss = 0.0
    running_real_loss = 0.0
    running_fake_loss = 0.0
    running_adv_loss = 0.0
    total = 0


    volumes = volumes.to(device)

    with torch.no_grad():

        B = volumes.shape[0]
        z = torch.randn(B, noise_dim, device=device) * 0.1


        #Generate our predicted values

        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))

        gen_outputs = gen_model(volumes, z)


        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
        #Post-Process
        pred_label = gen_outputs.squeeze(0).detach().cpu().numpy().argmax(axis=0)  # [Z,Y,X]
        Z, Y, X   = pred_label.shape

        # 2) Build pred_ids via watershed on each channel
        pred_ids = np.zeros((3, Z, Y, X), dtype=np.int32)
        for ch in (1, 2):
            mask = (pred_label == ch)

            # 1) Distance transform
            dist = distance_transform_edt(mask).astype(np.float32)


            # 2) Find peak coordinates (no 'indices' argument)
            peak_coords = peak_local_max(
                dist,
                min_distance=2,
                footprint=np.ones((3,3,3)),
                labels=mask
            )
            # peak_coords is an array of shape (N_peaks, 3)

            # 3) Build a markers image: each peak becomes a unique integer label
            markers = np.zeros_like(dist, dtype=np.int32)
            for idx, (z,y,x) in enumerate(peak_coords, start=1):
                markers[z, y, x] = idx

            # 4) Run watershed
            labels_ws = watershed(-dist, markers, mask=mask)
            pred_ids[ch] = labels_ws
        # 3) Match against GT as before
        all_matches = []
        for ch in [1, 2]:
            # build dict of predicted instance masks
            pred_instances = {
                (ch, inst_id): (pred_ids[ch] == inst_id)
                for inst_id in range(1, int(pred_ids[ch].max()) + 1)
            }

            gt_keys   = [k for k in gt_instances if k[0] == ch]
            pred_keys = list(pred_instances)

            cost = np.zeros((len(gt_keys), len(pred_keys)), dtype=np.float32)
            for i, gk in enumerate(gt_keys):
                gmask = gt_instances[gk]
                for j, pk in enumerate(pred_keys):
                    pmask = pred_instances[pk]
                    inter = np.logical_and(gmask, pmask).sum()
                    union = np.logical_or(gmask, pmask).sum()
                    cost[i, j] = - inter/(union + 1e-9)

            row_idx, col_idx = linear_sum_assignment(cost)
            for r, c in zip(row_idx, col_idx):
                gt_key   = gt_keys[r]
                pred_key = pred_keys[c]
                all_matches.append((gt_key, pred_key, -cost[r, c]))

        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
        return pred_label, all_matches, pred_ids



    total += volumes.size(0)


    print("Current_gen:", running_gen_loss/total)
    print("Current_real:", running_real_loss/total)
    print("Current_fake:", running_fake_loss/total)
    print("Current_adv:", running_adv_loss/total)
    print("Current:", running_loss/total)

    return running_loss / total

def buildPiff(pred_label, all_matches, pred_ids, name, id):
    rows = []
    Z, Y, X = pred_label.shape
    for (ch, gt_id), (ch2, pred_inst), iou in all_matches:
        assert ch == ch2
        mask = (pred_label == ch) & (pred_ids[ch] == pred_inst)
        zs, ys, xs = np.where(mask)
        cell_type = "body" if ch == 1 else "wall"
        for z, y, x in zip(zs, ys, xs):
            rows.append({
                "CellID":   int(gt_id),
                "CellType": cell_type,
                "x1":       int(x),
                "x2":       int(x),
                "y1":       int(y),
                "y2":       int(y),
                "z1":       int(z),
                "z2":       int(z),
            })

    # turn into a DataFrame
    df = pd.DataFrame(rows, columns=["CellID","CellType","x1","x2","y1","y2","z1","z2"])

    # sort by x1, then y1, then z1
    df = df.sort_values(by=["x1", "y1", "z1"], ascending=[True, True, True])

    # write out without index or header
    df.to_csv(f"{name}{id}.piff", sep=' ', index=False, header=False)



def getInferenceData(name, id):
    path = "C:\\Users\\evans\\Desktop\\Independent Study\\outputs\\" + name + id + ".piff"

    # parse to numpy volume, convert to tensor with channel dim
    inputVol = parse_voxel_file(path)
    inputTensor = torch.from_numpy(inputVol) #.unsqueeze(0)  # shape [1, D, H, W]
    
    #if self.transform:
     #   inputTensor = self.transform(inputTensor)

    IDVol = parse_voxel_file_for_ID_matching(path)
    IDTensor = torch.from_numpy(IDVol) #.unsqueeze(0)  # shape [1, D, H, W]

    gt_instances = {}
    # channels: 0=medium (ignore), 1=body, 2=wall
    for ch in [1, 2]:
        # grab all the IDs in this channel (background is encoded as 0)
        ids = np.unique(IDTensor[ch])
        ids = ids[ids != 0]   # drop the 0 background
        for id_ in ids:
            # make a boolean mask for that specific cell (or wall)
            mask = (IDTensor[ch] == id_)
            gt_instances[(ch, id_)] = mask

    return inputTensor.unsqueeze(0), gt_instances


def main():
     # User-defined parameters (adjust to your data)
    data_folder = "C:\\Users\\evans\\Desktop\\Independent Study\\outputs"
    latent_dim  = 256               # size of VAE bottleneck
    base_ch     = 32                # number of filters in first layer
    batch_size  = 2                 # samples per GPU batch
    epochs      = 100               # training duration
    gen_lr          = 1e-4              # learning rate
    disc_lr          = 1.5e-6              # learning rate






    # Setup PyTorch device and data loader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = VoxelDataset(data_folder)

    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True
    )



    # Instantiate model and begin training
    gen_model = UNet3D().to(device)
    disc_model = Discriminator3D().to(device)

    Loading = True
    runMode = 3 #1 Train, 2 Evaluate, 3 infer
    if Loading:
        gen_state  = torch.load("gen_check/unet3d_vae_checkpoint1094.pth",  map_location=device)
        disc_state = torch.load("disc_check/cnn3d_vae_checkpoint1094.pth", map_location=device)

        gen_model.load_state_dict(gen_state)
        disc_model.load_state_dict(disc_state)


    gen_optimizer = torch.optim.Adam(gen_model.parameters(), lr=gen_lr)
    disc_optimizer = torch.optim.Adam(disc_model.parameters(), lr=disc_lr)
    gen_criterion = nn.CrossEntropyLoss()
    disc_criterion = nn.BCEWithLogitsLoss()

    num_epochs = 2500

    for epoch in range(num_epochs):
        if epoch <= 110:
            epoch = 111

        with open("log.txt", "a") as f:
            f.write("Starting epoch " + str(epoch) + "\n")
        if runMode == 1:
            train(gen_model, disc_model, loader, gen_optimizer, disc_optimizer, gen_criterion, disc_criterion, device)
                        # Save weights for later use
            torch.save(gen_model.state_dict(), "gen_check/unet3d_vae_checkpoint" + str(epoch) + ".pth")
            torch.save(disc_model.state_dict(), "disc_check/cnn3d_vae_checkpoint" + str(epoch) + ".pth")
            print("Checkpoint saved.")
        elif runMode == 2:
            evaluate(gen_model, disc_model, loader, gen_optimizer, disc_optimizer, gen_criterion, disc_criterion, device)
        elif runMode == 3:
            name = "10_24Simulation"
            id = "000"
            volumes, gt_instances = getInferenceData(name, id)
            pred_label, all_matches, pred_ids = runinference(gen_model, volumes, gt_instances, device)
            name = "6_25Simulation"
            id = f"{(int(id) + 1):03d}"
            buildPiff(pred_label, all_matches, pred_ids, name, id)

        
        print("^^^^^^^^^^^^^^ Epoch ", epoch, "^^^^^^^^^^^")
        print("------------------------------")




if __name__ == "__main__":
   main()
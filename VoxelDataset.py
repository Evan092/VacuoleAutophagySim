import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from Utils import parse_voxel_file

class VoxelDataset(Dataset):
    """
    PyTorch Dataset for loading 3D voxel volumes.  
    Returns (input, target) pairs where target == input for autoencoder training.
    """
    def __init__(self, folder, transform=None):
        # collect all .txt or .piff files in folder
        self.paths = [
            os.path.join(folder, d)
            for d in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, d))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, key):
        #idx, outIdx = key
        idx, outputNumber, startStep, nSteps = key

        # parse to numpy volume, convert to tensor with channel dim


        output = f"outputs_{outputNumber:02d}"
        inputVol = parse_voxel_file(self.paths[idx] + "\\" + output + f"\\output{startStep:03d}.piff")
        inputTensor = torch.from_numpy(inputVol) #.unsqueeze(0)  # shape [1, D, H, W]
        if self.transform:
            inputTensor = self.transform(inputTensor)

        IDVol = parse_voxel_file(self.paths[idx] + "\\" + output + f"\\output{startStep:03d}.piff")
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


        outputVol = parse_voxel_file(self.paths[idx] + "\\" + output + f"\\output{(startStep+nSteps):03d}.piff")
        targetTensor = torch.from_numpy(outputVol)#.unsqueeze(0)  # shape [1, D, H, W]
        if self.transform:
            targetTensor = self.transform(targetTensor)


        xFlip = random.randint(0,1)
        yFlip = random.randint(0,1)
        zFlip = random.randint(0,1)

        flips = []

        if xFlip == 1:
            flips.append(1)
        if yFlip == 1:
            flips.append(2)
        if zFlip == 1:
            flips.append(3)

        inputTensor = torch.flip(inputTensor, dims=flips)
        targetTensor = torch.flip(targetTensor, dims=flips)

        return inputTensor, targetTensor, nSteps, gt_instances

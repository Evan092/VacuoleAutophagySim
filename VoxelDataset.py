import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from Utils import parse_voxel_file, parse_voxel_file_for_distance,voxel_points_to_self

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
            if (os.path.isdir(os.path.join(folder, d)) 
                and os.path.isdir(os.path.join(folder, d, "outputs_01"))
                and os.path.isdir(os.path.join(folder, d, "outputs_02"))
                and os.path.isdir(os.path.join(folder, d, "outputs_03"))
                and os.path.isdir(os.path.join(folder, d, "outputs_04"))
                and os.path.isdir(os.path.join(folder, d, "outputs_05"))
                and os.path.isfile(os.path.join(folder, d, "outputs_05", "output300.piff"))
                and not str(d).lower().endswith(".bat"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, key):
        #idx, outIdx = key
        idx, outputNumber, startStep, nSteps = key

        # parse to numpy volume, convert to tensor with channel dim


        output = f"outputs_{outputNumber:02d}"

        inputTensor,_ = parse_voxel_file_for_distance(self.paths[idx] + "\\" + output + f"\\output{startStep:03d}.piff")

        inputTensor = inputTensor.squeeze(0)

        #if (not voxel_points_to_self(inputTensor,100,100,100)):
         #   raise Exception
        # channels: 0=medium (ignore), 1=body, 2=wall
        #for ch in [1, 2]:
            # grab all the IDs in this channel (background is encoded as 0)
            #ids = np.unique(inputTensor[ch])
            #ids = ids[ids != 0]   # drop the 0 background
            #for id_ in ids:
                # make a boolean mask for that specific cell (or wall)
                #mask = (inputTensor[ch] == id_)
                #gt_instances[(ch, id_)] = mask

        #IDTensor = inputTensor.clone().detach()

        #inputTensor = torch.from_numpy(inputVol) #.unsqueeze(0)  # shape [1, D, H, W]
        if self.transform:
            inputTensor = self.transform(inputTensor)

        #IDTensor = parse_voxel_file(self.paths[idx] + "\\" + output + f"\\output{startStep:03d}.piff")
        #IDTensor = torch.from_numpy(IDVol) #.unsqueeze(0)  # shape [1, D, H, W]




        targetTensor,_  = parse_voxel_file_for_distance(self.paths[idx] + "\\" + output + f"\\output{(startStep+nSteps):03d}.piff")
        targetTensor = targetTensor.squeeze(0)

        #if (not voxel_points_to_self(targetTensor,100,100,100)):
            #raise Exception

        if self.transform:
            targetTensor = self.transform(targetTensor)

        zFlip = random.randint(0,1)
        xFlip = random.randint(0,1)
        yFlip = random.randint(0,1)

        flips = []

        if zFlip == 1:
            inputTensor[0] = inputTensor[0]*-1
            targetTensor[0] = targetTensor[0]*-1
            flips.append(1)
        if yFlip == 1:
            inputTensor[1] = inputTensor[1]*-1
            targetTensor[1] = targetTensor[1]*-1
            flips.append(2)
        if xFlip == 1:
            inputTensor[2] = inputTensor[2]*-1
            targetTensor[2] = targetTensor[2]*-1
            flips.append(3)

        inputTensor = torch.flip(inputTensor, dims=flips)
        targetTensor = torch.flip(targetTensor, dims=flips)

        return inputTensor.clone(), targetTensor.clone(), nSteps, (self.paths[idx] + "\\" + output + f"\\output{startStep:03d}.piff"),(self.paths[idx] + "\\" + output + f"\\output{(startStep+nSteps):03d}.piff")

import numpy as np
import torch
from VacuoleAutophagySim.Utils import parse_voxel_file
from torch.utils.data import Dataset


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

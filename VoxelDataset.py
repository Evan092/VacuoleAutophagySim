import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from Constants import MAX_VOXEL_DIM
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

    def __getitem__Old(self, key):
        #idx, outIdx = key
        idx, outputNumber, startStep, nSteps = key

        # parse to numpy volume, convert to tensor with channel dim


        output = f"outputs_{outputNumber:02d}"

        inputTensor,inputVol, oldCenters = parse_voxel_file_for_distance(self.paths[idx] + "\\" + output + f"\\output{startStep:03d}.piff")

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



        outputNumber=1
        output = f"outputs_{outputNumber:02d}"
        targetTensor1,vol1, centers1  = parse_voxel_file_for_distance(self.paths[idx] + "\\" + output + f"\\output{(startStep+nSteps):03d}.piff")
        targetTensor1 = targetTensor1.squeeze(0)

        outputNumber=2
        output = f"outputs_{outputNumber:02d}"
        targetTensor2,vol2,centers2  = parse_voxel_file_for_distance(self.paths[idx] + "\\" + output + f"\\output{(startStep+nSteps):03d}.piff")
        targetTensor2 = targetTensor2.squeeze(0)

        outputNumber=3
        output = f"outputs_{outputNumber:02d}"
        targetTensor3,vol3,centers3  = parse_voxel_file_for_distance(self.paths[idx] + "\\" + output + f"\\output{(startStep+nSteps):03d}.piff")
        targetTensor3 = targetTensor3.squeeze(0)

        outputNumber=4
        output = f"outputs_{outputNumber:02d}"
        targetTensor4,vol4,centers4  = parse_voxel_file_for_distance(self.paths[idx] + "\\" + output + f"\\output{(startStep+nSteps):03d}.piff")
        targetTensor4 = targetTensor4.squeeze(0)

        outputNumber=5
        output = f"outputs_{outputNumber:02d}"
        targetTensor5,vol5,centers5  = parse_voxel_file_for_distance(self.paths[idx] + "\\" + output + f"\\output{(startStep+nSteps):03d}.piff")
        targetTensor5 = targetTensor5.squeeze(0)




        #if (not voxel_points_to_self(targetTensor,100,100,100)):
            #raise Exception

        if self.transform:
            targetTensor1 = self.transform(targetTensor1)
            targetTensor2 = self.transform(targetTensor2)
            targetTensor3 = self.transform(targetTensor3)
            targetTensor4 = self.transform(targetTensor4)
            targetTensor5 = self.transform(targetTensor5)

        zFlip = random.randint(0,1)
        xFlip = random.randint(0,1)
        yFlip = random.randint(0,1)

        flips = []

        if zFlip == 1:
            inputTensor[0] = inputTensor[0]*-1
            targetTensor1[0] = targetTensor1[0]*-1
            targetTensor2[0] = targetTensor2[0]*-1
            targetTensor3[0] = targetTensor3[0]*-1
            targetTensor4[0] = targetTensor4[0]*-1
            targetTensor5[0] = targetTensor5[0]*-1
            oldCenters[:,1] = MAX_VOXEL_DIM - oldCenters[:,1]
            centers1[:,1] = MAX_VOXEL_DIM - centers1[:,1]
            centers2[:,1] = MAX_VOXEL_DIM - centers2[:,1]
            centers3[:,1] = MAX_VOXEL_DIM - centers3[:,1]
            flips.append(1)
        if yFlip == 1:
            inputTensor[1] = inputTensor[1]*-1
            targetTensor1[1] = targetTensor1[1]*-1
            targetTensor2[1] = targetTensor2[1]*-1
            targetTensor3[1] = targetTensor3[1]*-1
            targetTensor4[1] = targetTensor4[1]*-1
            targetTensor5[1] = targetTensor5[1]*-1
            oldCenters[:,2] = MAX_VOXEL_DIM - oldCenters[:,2]
            centers1[:,2] = MAX_VOXEL_DIM - centers1[:,2]
            centers2[:,2] = MAX_VOXEL_DIM - centers2[:,2]
            centers3[:,2] = MAX_VOXEL_DIM - centers3[:,2]
            flips.append(2)
        if xFlip == 1:
            inputTensor[2] = inputTensor[2]*-1
            targetTensor1[2] = targetTensor1[2]*-1
            targetTensor2[2] = targetTensor2[2]*-1
            targetTensor3[2] = targetTensor3[2]*-1
            targetTensor4[2] = targetTensor4[2]*-1
            targetTensor5[2] = targetTensor5[2]*-1
            oldCenters[:,3] = MAX_VOXEL_DIM - oldCenters[:,3]
            centers1[:,3] = MAX_VOXEL_DIM - centers1[:,3]
            centers2[:,3] = MAX_VOXEL_DIM - centers2[:,3]
            centers3[:,3] = MAX_VOXEL_DIM - centers3[:,3]
            flips.append(3)

        inputTensor = torch.flip(inputTensor, dims=flips)
        targetTensor1 = torch.flip(targetTensor1, dims=flips)
        targetTensor2 = torch.flip(targetTensor2, dims=flips)
        targetTensor3 = torch.flip(targetTensor3, dims=flips)
        targetTensor4 = torch.flip(targetTensor4, dims=flips)
        targetTensor5 = torch.flip(targetTensor5, dims=flips)
        inputVol = torch.flip(inputVol, dims=flips)
        vol1 = torch.flip(vol1, dims=flips)
        vol2 = torch.flip(vol2, dims=flips)
        vol3 = torch.flip(vol3, dims=flips)
        vol4 = torch.flip(vol4, dims=flips)
        vol5 = torch.flip(vol5, dims=flips)

        assert len(oldCenters) == len(centers1) == len(centers2) == len(centers3) == len(centers4) == len(centers5)

        assert (torch.equal(torch.unique(inputVol), torch.unique(vol1)) and torch.equal(torch.unique(inputVol), torch.unique(vol2)) and torch.equal(torch.unique(inputVol), torch.unique(vol3)) and torch.equal(torch.unique(inputVol), torch.unique(vol4)) and torch.equal(torch.unique(inputVol), torch.unique(vol5)))
        
        assert torch.all(oldCenters[:, 0] == torch.arange(1, oldCenters.shape[0] + 1, device=oldCenters.device, dtype=oldCenters.dtype))
        assert torch.all(centers1[:, 0] == torch.arange(1, centers1.shape[0] + 1, device=centers1.device, dtype=centers1.dtype))
        assert torch.all(centers2[:, 0] == torch.arange(1, centers2.shape[0] + 1, device=centers2.device, dtype=centers2.dtype))
        assert torch.all(centers3[:, 0] == torch.arange(1, centers3.shape[0] + 1, device=centers3.device, dtype=centers3.dtype))
        assert torch.all(centers4[:, 0] == torch.arange(1, centers4.shape[0] + 1, device=centers4.device, dtype=centers4.dtype))
        assert torch.all(centers5[:, 0] == torch.arange(1, centers5.shape[0] + 1, device=centers5.device, dtype=centers5.dtype))


        return (inputTensor.clone(), oldCenters, inputVol), (targetTensor1.clone(),centers1, vol1),(targetTensor2.clone(),centers2, vol2),(targetTensor3.clone(),centers3, vol3),(targetTensor4.clone(),centers4, vol4),(targetTensor5.clone(),centers5, vol5), nSteps, (self.paths[idx] + "\\" + output + f"\\output{startStep:03d}.piff")


    def __getitem__(self, key):
        # key gives us:
        # idx          -> which sim folder in self.paths
        # outputNumber -> which outputs_## dir to use for the *input* state
        # startStep    -> which timestep to grab for the input
        # nSteps       -> how far ahead to sample targets
        idx, outputNumber, startStep, nSteps = key

        # Build the path for the "current" state (volumes, oldCenters, inputVol in your old code)
        # This used to feed (inputTensor, inputVol, oldCenters)
        input_output_dir = f"outputs_{outputNumber:02d}"
        oldPath = (
            self.paths[idx]
            + "\\"
            + input_output_dir
            + f"\\output{startStep:03d}.piff"
        )

        # Build the 5 future target paths (targetTensor1..5, vol1..5, centers1..5)
        # These were always pulled from outputs_01 .. outputs_05 at (startStep + nSteps)
        future_step = startStep + nSteps

        t1Path = (
            self.paths[idx]
            + "\\"
            + "outputs_01"
            + f"\\output{future_step:03d}.piff"
        )

        t2Path = (
            self.paths[idx]
            + "\\"
            + "outputs_02"
            + f"\\output{future_step:03d}.piff"
        )

        t3Path = (
            self.paths[idx]
            + "\\"
            + "outputs_03"
            + f"\\output{future_step:03d}.piff"
        )

        t4Path = (
            self.paths[idx]
            + "\\"
            + "outputs_04"
            + f"\\output{future_step:03d}.piff"
        )

        t5Path = (
            self.paths[idx]
            + "\\"
            + "outputs_05"
            + f"\\output{future_step:03d}.piff"
        )

        # ---------------------------------
        # flips logic (EXACTLY your logic)
        # ---------------------------------
        zFlip = random.randint(0,1)
        xFlip = random.randint(0,1)
        yFlip = random.randint(0,1)

        flips = []
        # you were appending 1 for zFlip, 2 for yFlip, 3 for xFlip in that order
        if zFlip == 1:
            flips.append(1)
        else:
            flips.append(0)

        if yFlip == 1:
            flips.append(2)
        else:
            flips.append(0)

        if xFlip == 1:
            flips.append(3)
        else:
            flips.append(0)

        flips = torch.tensor(flips)
        # We are NOT applying flips here (no tensors are loaded here anymore).
        # We just return the flip instructions so the training loop / loadData
        # can apply them after loading.

        # ---------------------------------
        # RETURN
        # ---------------------------------
        # Your train loop expects:
        # for index, ((oldPath), (t1Path), (t2Path), (t3Path), (t4Path), (t5Path), steps) in enumerate(dataloader):
        #
        # We also need flips now, so return each path bundled with flips.
        # And your loop's "steps" argument was `nSteps`.
        #
        # Structure:
        #   ( (oldPath, flips),
        #     (t1Path, flips),
        #     (t2Path, flips),
        #     (t3Path, flips),
        #     (t4Path, flips),
        #     (t5Path, flips),
        #     nSteps )
        #
        # That way nothing big is touched here.

        return (
            (oldPath),
            (t1Path),
            (t2Path),
            (t3Path),
            (t4Path),
            (t5Path),
            flips,
            nSteps
        )

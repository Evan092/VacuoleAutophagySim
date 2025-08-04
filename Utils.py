
from datetime import datetime
import os
import numpy as np
import torch


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




import numpy as np
from scipy.ndimage import zoom

import numpy as np
from scipy.ndimage import zoom

import numpy as np
from scipy.ndimage import zoom

def pad_crop_resize_2C(vol, target_size=200):
    """
    vol: np.ndarray, shape (2, D, H, W)
         channels correspond to your two masks; both 0 means background.
    returns: vol padded/cropped/resized to spatial dims == target_size,
             cropping out as many all-zero voxels as possible while
             preserving any nonzero in either channel.
    """
    C, D, H, W = vol.shape
    assert C == 2, "Expected 2 channels"
    t = target_size

    # Step 1: Pad any dim < t (same as before, channel-agnostic)
    pad_z = max(t - D, 0)
    pad_y = max(t - H, 0)
    pad_x = max(t - W, 0)
    pzb, pza = pad_z // 2, pad_z - pad_z // 2
    pyb, pya = pad_y // 2, pad_y - pad_y // 2
    pxb, pxa = pad_x // 2, pad_x - pad_x // 2

    vol = np.pad(
        vol,
        ((0, 0), (pzb, pza), (pyb, pya), (pxb, pxa)),
        mode='constant',
        constant_values=0,
    )

    # Step 2: Crop along each axis, only removing slices that are
    # all zero **across both channels**
    def crop_axis(mask3d, axis):
        # mask3d: shape (D,H,W), True where any channel≠0
        nonzero = np.any(mask3d, axis=tuple(i for i in range(mask3d.ndim) if i != axis))
        idxs = np.where(nonzero)[0]
        if len(idxs) == 0:
            return slice(None)
        first, last = idxs[0], idxs[-1]
        L = mask3d.shape[axis]
        excess = L - t
        if excess <= 0:
            return slice(None)

        zeros_before = first
        zeros_after  = L - 1 - last
        if zeros_before + zeros_after >= excess:
            rem_before = min(zeros_before, excess // 2)
            rem_after  = excess - rem_before
            start = rem_before
            end   = L - rem_after
            return slice(start, end)
        else:
            return slice(None)

    # build a combined occupancy mask
    combined = np.any(vol != 0, axis=0)  # shape (D,H,W)
    sz = crop_axis(combined, axis=0)
    sy = crop_axis(combined, axis=1)
    sx = crop_axis(combined, axis=2)
    vol = vol[:, sz, sy, sx]

    # Step 3: Resize (nearest‐neighbor) if still not target_size
    _, D2, H2, W2 = vol.shape
    if not (D2 == H2 == W2 == t):
        factors = (1, t / D2, t / H2, t / W2)
        vol = zoom(vol, zoom=factors, order=0)

    return vol.astype(np.float32)


def pad_crop_resize(vol, target_size=200):
    """
    vol: np.ndarray, shape (1, D, H, W)
    returns: vol padded/cropped/resized to spatial dims == target_size,
             cropping out as many 0s as possible but keeping all 1s and 2s
    """
    C, D, H, W = vol.shape
    assert C == 1
    t = target_size

    # Step 1: Pad any dim < t
    pad_z = max(t - D, 0)
    pad_y = max(t - H, 0)
    pad_x = max(t - W, 0)

    pzb, pza = pad_z // 2, pad_z - pad_z // 2
    pyb, pya = pad_y // 2, pad_y - pad_y // 2
    pxb, pxa = pad_x // 2, pad_x - pad_x // 2

    vol = np.pad(
        vol,
        (
            (0, 0),
            (pzb, pza),
            (pyb, pya),
            (pxb, pxa),
        ),
        mode='constant',
        constant_values=0,
    )

    # Step 2: Crop excess only from zero or 3-valued regions (keep all 1s and 2s)
    def crop_axis(data, axis):
        # 1s and 2s must be preserved
        preserve = (data == 1) | (data == 2)
        nonzero = np.any(preserve, axis=tuple(i for i in range(data.ndim) if i != axis))
        first, last = np.where(nonzero)[0][[0, -1]]
        L = data.shape[axis]
        excess = L - t
        if excess <= 0:
            return slice(None)

        zeros_before = first
        zeros_after = L - 1 - last

        if zeros_before + zeros_after >= excess:
            # Remove as many 3s as possible from outer regions
            remove_before = min(zeros_before, excess // 2)
            remove_after = excess - remove_before
            start = remove_before
            end = L - remove_after
            return slice(start, end)
        else:
            # Not enough removable space without hitting 1s or 2s
            return slice(None)

    sz = crop_axis(vol[0], axis=0)
    sy = crop_axis(vol[0], axis=1)
    sx = crop_axis(vol[0], axis=2)

    vol = vol[:, sz, sy, sx]

    # Step 3: Resize if needed
    _, D2, H2, W2 = vol.shape
    if not (D2 == H2 == W2 == t):
        factors = (1, t / D2, t / H2, t / W2)
        vol = zoom(vol, zoom=factors, order=0)

    return vol.astype(np.float32)



loadExisting = True
delete = False
saveLoaded = True


def parse_voxel_file(path):
    """
    Read a text-based voxel file where each line specifies a solid block:
      ID Type x1 x2 y1 y2 z1 z2
    Build and return a one-hot (3, D, H, W) volume.
    """
    voxels = []
    max_x = max_y = max_z = 0

    if loadExisting and os.path.isfile(os.path.join(str(path).removesuffix(".piff") + ".pt")):
        tensor = torch.load(os.path.join(str(path).removesuffix(".piff") + ".pt"))
        if tensor.shape[1] != 200:
            os.remove(os.path.join(str(path).removesuffix(".piff") + ".pt"))
        else:
            body_mask = (tensor == 1).float()  # 1s where Body
            wall_mask = (tensor == 2).float()  # 1s where Wall

            # Stack into 2-channel tensor: [2, D, H, W]
            tensor = torch.cat([body_mask, wall_mask], dim=0)

            return tensor
    elif delete and os.path.isfile(os.path.join(str(path).removesuffix(".piff") + ".pt")):
        os.remove(os.path.join(str(path).removesuffix(".piff") + ".pt"))

    # 1) Parse file
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
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)
            max_z = max(max_z, z2)

    # 2) Build initial volume
    vol = np.zeros((1, max_z + 1, max_y + 1, max_x + 1), dtype=np.float32)
    for cell_type, x1, x2, y1, y2, z1, z2 in voxels:
        if cell_type == "Body":
            vol[0, z1 : z2 + 1, y1 : y2 + 1, x1 : x2 + 1] = 1.0
        elif cell_type == "Wall":
            vol[0, z1 : z2 + 1, y1 : y2 + 1, x1 : x2 + 1] = 2.0
        else:
            vol[0, z1 : z2 + 1, y1 : y2 + 1, x1 : x2 + 1] = 0.0

    vol = pad_crop_resize(vol)

    tensor = torch.from_numpy(vol)

    if saveLoaded:
        torch.save(tensor, os.path.join(str(path).removesuffix(".piff") + ".pt"))

    # Assume tensor has shape [1, D, H, W]
    body_mask = (tensor == 1).float()  # 1s where Body
    wall_mask = (tensor == 2).float()  # 1s where Wall
    Neither = (tensor == 0).float()  # 1s where Wall

    # Stack into 2-channel tensor: [2, D, H, W]
    tensor = torch.cat([body_mask, wall_mask, Neither], dim=0)

    return tensor






def parse_voxel_fileOG(path):
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

    vol = pad_crop_resize(vol)

    return vol

def parse_voxel_file_for_ID_matching2(path):
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

def parse_voxel_file_for_ID_matching(path):
    """
    Read a text-based voxel file where each line specifies a solid block:
      ID Type x1 x2 y1 y2 z1 z2
    Build and return a one-hot (3, D, H, W) volume.
    """
    voxels = []
    max_x = max_y = max_z = 0

    # 1) Parse file
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
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)
            max_z = max(max_z, z2)

    # 2) Build initial volume
    vol = np.zeros((2, max_z + 1, max_y + 1, max_x + 1), dtype=np.float32)
    for cell_ID, cell_type, x1, x2, y1, y2, z1, z2 in voxels:
        if cell_type == "Body":
            vol[0, z1 : z2 + 1, y1 : y2 + 1, x1 : x2 + 1] = cell_ID
        elif cell_type == "Wall":
            vol[1, z1 : z2 + 1, y1 : y2 + 1, x1 : x2 + 1] = cell_ID

    vol = pad_crop_resize_2C(vol)

    tensor = torch.from_numpy(vol)

    return tensor

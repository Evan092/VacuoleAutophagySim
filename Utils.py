
from datetime import datetime
import heapq
from math import floor
import os
from time import perf_counter
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt, zoom
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from Constants import MAX_VOXEL_DIM

loadExisting = True
delete = False
saveLoaded = True

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
            Neither = (tensor == 0).float()

            # Stack into 2-channel tensor: [2, D, H, W]
            tensor = torch.cat([body_mask, wall_mask,Neither], dim=0)

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
    tensor = np.flip(tensor, axis=3)
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



#NEW BELOW

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

    #vol = np.pad(
    #    vol,
    #    (
    #        (0, 0),
    #        (pzb, pza),
    #        (pyb, pya),
    #        (pxb, pxa),
    #    ),
    #    mode='constant',
    #    constant_values=0,
    #)

    # Step 2: Crop excess only from -1 valued regions (keep all 1s and 2s)
    def crop_axis(data, axis):
        preserve = (data != 1)
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




# CHUNK 1 — drop-in parallel EDT maxima per id
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import distance_transform_edt

def _edt_max_worker(args):
    cid, mask_roi, z0, y0, x0 = args
    if mask_roi.size == 0 or not mask_roi.any():
        return None
    dist = distance_transform_edt(mask_roi)
    idx = int(dist.argmax())
    z, y, x = np.unravel_index(idx, dist.shape)
    return (int(cid), int(z0 + z), int(y0 + y), int(x0 + x))

def edt_maxima_per_id(vol, workers=None, parallel=False):
    """
    vol: (D,H,W) float32 with IDs (>0) for solids; others <= 0
    Return: centers = [(id, z, y, x), ...] EDT argmax per unique positive ID
    (drop-in replacement; parallel per-ID, ROI-cropped for less data shipped)
    """
    mask_pos = vol > 0
    if not mask_pos.any():
        return []

    ids = np.unique(vol[mask_pos]).astype(np.int64)

    if (not parallel):
        mask_pos = vol > 0
        if not mask_pos.any():
            return [], []

        ids = np.unique(vol[mask_pos]).astype(np.int64)
        centers = []

        for cid in ids:
            mask = (vol == cid)
            if not mask.any():
                continue
            dist = distance_transform_edt(mask)
            z, y, x = np.unravel_index(dist.argmax(), dist.shape)
            centers.append((int(cid), int(z), int(y), int(x)))

        return centers

    jobs = []
    for cid in ids:
        mask = (vol == cid)
        if not mask.any():
            continue
        # crop to tight ROI to reduce work/IPC
        zz, yy, xx = mask.nonzero()
        z0, y0, x0 = zz.min(), yy.min(), xx.min()
        z1, y1, x1 = zz.max() + 1, yy.max() + 1, xx.max() + 1
        mask_roi = mask[z0:z1, y0:y1, x0:x1]
        jobs.append((int(cid), mask_roi, int(z0), int(y0), int(x0)))

    if not jobs:
        return []

    workers = workers or os.cpu_count() or 4
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_edt_max_worker, jobs, chunksize=1))

    centers = [r for r in results if r is not None]
    return centers









def edt_maxima_per_idOLD(vol):
    """
    vol: (D, H, W) float32 with IDs (>0) for solids; others <=0
    Return:
      centers: list of (z,y,x) EDT-argmax per unique positive ID
      xs: sorted unique x's among centers
    """
    mask_pos = vol > 0
    if not mask_pos.any():
        return [], []

    ids = np.unique(vol[mask_pos]).astype(np.int64)
    centers = []

    for cid in ids:
        mask = (vol == cid)
        if not mask.any():
            continue
        dist = distance_transform_edt(mask)
        z, y, x = np.unravel_index(dist.argmax(), dist.shape)
        centers.append((int(cid), int(z), int(y), int(x)))

    return centers


def unit_dir_and_distance_zyx(coords, spacing=(1.0, 1.0, 1.0), distance_units="vox"):
    """
    LAYOUT: Z, Y, X  (fastest axis = X)

    coords: torch.Tensor shape (3, Z, Y, X), storing [Xc, Yc, Zc] per voxel.
            Invalid voxels are (-1, -1, -1).
    spacing: (dx, dy, dz) in X,Y,Z order.
    distance_units: 'vox' or 'phys'
    Returns: (4, Z, Y, X) float32 = [ux, uy, uz, r]
      - u is the unit vector from voxel (x,y,z) to center (Xc,Yc,Zc)
      - r is the distance (vox or phys)
      - invalid -> u=(0,0,0), r=-1
    """
    assert isinstance(coords, torch.Tensor), "coords must be a torch.Tensor"
    dt = torch.float32
    C = coords.to(dtype=dt)
    _, Z, Y, X = C.shape
    dev = C.device

    # index grids in Z,Y,X
    zz = torch.arange(Z, dtype=dt, device=dev).view(Z, 1, 1).expand(Z, Y, X)
    yy = torch.arange(Y, dtype=dt, device=dev).view(1, Y, 1).expand(Z, Y, X)
    xx = torch.arange(X, dtype=dt, device=dev).view(1, 1, X).expand(Z, Y, X)

    invalid = (C[0] == -1) & (C[1] == -1) & (C[2] == -1)

    # voxel → center offsets (always X,Y,Z semantics for the values)
    dx = (C[2] - xx).masked_fill(invalid, 0.0)
    dy = (C[1] - yy).masked_fill(invalid, 0.0)
    dz = (C[0] - zz).masked_fill(invalid, 0.0)

    if distance_units == "phys":
        sz, sy, sx= torch.tensor(spacing, dtype=dt, device=dev)  # (dx,dy,dz)
        vz, vy, vx = dz * sz, dy * sy, dx * sx
    else:
        vz, vy, vx = dz, dy, dx

    r = torch.sqrt(vz * vz + vy * vy + vx * vx)
    denom = torch.clamp(r, min=1e-8)
    r= r / MAX_VOXEL_DIM
    uz, uy, ux = vz / denom, vy / denom, vx / denom

    # enforce sentinel on invalid
    uz = uz.masked_fill(invalid, 0.0)
    uy = uy.masked_fill(invalid, 0.0)
    ux = ux.masked_fill(invalid, 0.0)
    r  =  r.masked_fill(invalid, -1.0)

    return torch.stack([uz, uy, ux, r], dim=0).contiguous()

import torch

def get_voxel_center(targets: torch.Tensor, z: int, y: int, x: int, scale: float = MAX_VOXEL_DIM):
    """
    Compute the pointed-to center for a voxel.
    
    Parameters
    ----------
    targets : torch.Tensor
        Shape [B, 4, Z, Y, X].
        channels 0–2 = (dz, dy, dx), channel 3 = scaled distance.
    z, y, x : int
        Voxel coordinates.
    scale : float
        Factor used to unscale the distance (default=200.0).
    
    Returns
    -------
    (cz, cy, cx) : tuple of floats
        Continuous pointed center coordinates.
    """
    dz = targets[0, 0, z, y, x].item()
    dy = targets[0, 1, z, y, x].item()
    dx = targets[0, 2, z, y, x].item()
    dist = targets[0, 3, z, y, x].item() * scale

    cz = z + dz * dist
    cy = y + dy * dist
    cx = x + dx * dist
    return (cz, cy, cx)

@torch.no_grad()
def sum_with_next_from(
    flow_vals: torch.Tensor,           # evolving accumulator (e.g., tmp)
    flow_dirs: torch.Tensor,           # field used to compute destinations (e.g., smoothedDistance)
    avoid_self: bool = True,
    mask: Optional[torch.Tensor] = None,    # previous self-pointing mask (bool), or None
    neighbor_vals: Optional[torch.Tensor] = None,  # if set, gather neighbor from here
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each voxel i: out[i] = flow_vals[i] + neighbor_vals[ dest(i) ],
    where dest(i) = i + round(flow_dirs[i]) (nearest, clamped).

    Shapes: (3,D,H,W) or (N,3,D,H,W) for all tensors.

    Stops updating when:
      - A voxel points to itself on two consecutive calls (mask & self_now), OR
      - Its (rounded) destination would be out of bounds (immediate stop for that call), OR
      - The neighbor vector at dest has any NaN component (skip add to avoid propagating NaN).
      - (If avoid_self) skip add when dest == self.

    Returns:
      out, self_now
        out: summed tensor, same shape as flow_vals
        self_now: bool mask (N,D,H,W) or (D,H,W), True where dest==self (this call)
    """
    # ---- shape checks & batching ----
    assert flow_vals.shape == flow_dirs.shape and flow_vals.dim() in (4, 5), \
        "flow_vals and flow_dirs must have same shape: (3,D,H,W) or (N,3,D,H,W)"
    has_batch = (flow_dirs.dim() == 5)
    if neighbor_vals is None:
        neighbor_vals = flow_vals
    else:
        assert neighbor_vals.shape == flow_vals.shape, "neighbor_vals must match flow_vals shape"

    if not has_batch:
        flow_vals    = flow_vals.unsqueeze(0)
        flow_dirs    = flow_dirs.unsqueeze(0)
        neighbor_vals = neighbor_vals.unsqueeze(0)

    N, C, D, H, W = flow_vals.shape
    assert C == 3
    dev  = flow_vals.device

    # ---- validity of directions ----
    dz, dy, dx = flow_dirs[:, 0], flow_dirs[:, 1], flow_dirs[:, 2]
    mag   = torch.sqrt(dz*dz + dy*dy + dx*dx)
    valid = torch.isfinite(mag) & (mag > 0)

    # ---- grid coords ----
    z = torch.arange(D, device=dev, dtype=torch.float32)
    y = torch.arange(H, device=dev, dtype=torch.float32)
    x = torch.arange(W, device=dev, dtype=torch.float32)
    Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')
    Z = Z.unsqueeze(0).expand(N, -1, -1, -1)
    Y = Y.unsqueeze(0).expand(N, -1, -1, -1)
    X = X.unsqueeze(0).expand(N, -1, -1, -1)

    # ---- destination (rounded BEFORE clamp to detect OOB) ----
    RZ = torch.where(valid, Z + dz, Z).round()
    RY = torch.where(valid, Y + dy, Y).round()
    RX = torch.where(valid, X + dx, X).round()

    oob_now = (RZ < 0) | (RZ >= D) | (RY < 0) | (RY >= H) | (RX < 0) | (RX >= W)

    TZ = RZ.clamp(0, D-1).to(torch.long)
    TY = RY.clamp(0, H-1).to(torch.long)
    TX = RX.clamp(0, W-1).to(torch.long)

    ZL, YL, XL = Z.to(torch.long), Y.to(torch.long), X.to(torch.long)

    # ---- self detection (this call) ----
    self_now = (TZ == ZL) & (TY == YL) & (TX == XL)  # (N,D,H,W) bool

    # ---- previous self mask ----
    if mask is None:
        prev_self = torch.zeros_like(self_now, dtype=torch.int32, device=dev)
    else:
        prev_self = mask.to(device=dev, dtype=torch.int32)
        if prev_self.dim() == 3:  # allow (D,H,W)
            prev_self = prev_self.unsqueeze(0)
        assert prev_self.shape == self_now.shape

    # ---- stopping logic ----
    stop = (prev_self > 50) | oob_now

    eff_valid = valid & ~stop
    if avoid_self:
        eff_valid = eff_valid & ~self_now  # skip doubling

    # ---- gather neighbor from neighbor_vals (NOT from flow_vals) ----
    dest_lin = (TZ*H*W + TY*W + TX).view(N, -1)     # (N,P)
    idx      = dest_lin.unsqueeze(1).expand(N, C, -1)

    vals_flat   = flow_vals.view(N, C, -1)
    neigh_flat  = neighbor_vals.view(N, C, -1)
    next_flat   = torch.gather(neigh_flat, 2, idx)  # from neighbor_vals

    # ---- NaN-safe add: zero-out neighbors with any NaN component, and apply eff_valid ----
    neighbor_ok = torch.isfinite(next_flat).all(dim=1, keepdim=True)            # (N,1,P)
    add_mask    = eff_valid.view(N, 1, -1) & neighbor_ok                        # (N,1,P)
    add_mask_exp = add_mask.expand_as(next_flat)                                # (N,3,P)
    next_safe    = torch.where(add_mask_exp, next_flat, torch.zeros_like(next_flat))

    out_flat = vals_flat + next_safe
    out = out_flat.view(N, C, D, H, W)

    if not has_batch:
        out = out.squeeze(0)
        self_now = self_now.squeeze(0)

    return out, self_now+prev_self


@torch.no_grad()
def snap_vectors_to_nearest_non_nanV3(
    flow: torch.Tensor,
    search_radius: int = 1,
    keep_original_if_none: bool = True,
    max_chunk_voxels: int = 500_000,   # process bad targets in chunks to bound memory
) -> torch.Tensor:
    assert flow.ndim == 4 and flow.shape[0] == 3, "flow must be (3,D,H,W)"
    _, D, H, W = flow.shape
    dev, dtype = flow.device, flow.dtype

    # finiteness masks
    src_finite = torch.isfinite(flow).all(dim=0)     # (D,H,W)
    dest_finite = src_finite.view(-1)                # same criterion at destinations

    # grid
    z = torch.arange(D, device=dev, dtype=dtype)
    y = torch.arange(H, device=dev, dtype=dtype)
    x = torch.arange(W, device=dev, dtype=dtype)
    Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')

    dz, dy, dx = flow[0], flow[1], flow[2]

    # endpoints
    PZ = Z + dz
    PY = Y + dy
    PX = X + dx

    # safe endpoints (avoid casting NaN to long)
    PZs = torch.where(src_finite, PZ, Z)
    PYs = torch.where(src_finite, PY, Y)
    PXs = torch.where(src_finite, PX, X)

    # initial rounded targets
    TZ0 = torch.round(PZs).clamp_(0, D-1).to(torch.long)
    TY0 = torch.round(PYs).clamp_(0, H-1).to(torch.long)
    TX0 = torch.round(PXs).clamp_(0, W-1).to(torch.long)

    lin0 = (TZ0 * (H*W) + TY0 * W + TX0).view(-1)
    init_valid = dest_finite.gather(0, lin0).view(D, H, W)

    # fast path: everything valid
    if init_valid.all().item():
        snapped = torch.stack([
            TZ0.to(dtype) - Z,
            TY0.to(dtype) - Y,
            TX0.to(dtype) - X
        ], dim=0)
        return torch.where(src_finite.unsqueeze(0), snapped, flow)

    # prepare outputs
    best_Z = TZ0.clone()
    best_Y = TY0.clone()
    best_X = TX0.clone()
    best_found = init_valid.clone()

    # current best distance from safe endpoint to current best index
    best_dist2 = (PZs - TZ0.to(dtype))**2 + (PYs - TY0.to(dtype))**2 + (PXs - TX0.to(dtype))**2

    # index list of bad positions to limit work/memory
    bad_mask = ~init_valid
    bad_idx = bad_mask.nonzero(as_tuple=False)  # (N,3)
    N = bad_idx.size(0)
    if N == 0:
        snapped = torch.stack([
            best_Z.to(dtype) - Z,
            best_Y.to(dtype) - Y,
            best_X.to(dtype) - X
        ], dim=0)
        return torch.where(src_finite.unsqueeze(0), snapped, flow)

    # offsets within Chebyshev radius (excluding 0,0,0)
    rng = range(-search_radius, search_radius + 1)
    offsets = [(oz, oy, ox) for oz in rng for oy in rng for ox in rng
               if not (oz == 0 and oy == 0 and ox == 0)]

    # process bad voxels in chunks to bound memory
    s = 0
    while s < N:
        e = min(s + max_chunk_voxels, N)
        idx_chunk = bad_idx[s:e]                  # (M,3)
        zz, yy, xx = idx_chunk[:,0], idx_chunk[:,1], idx_chunk[:,2]

        # gather per-voxel values for this chunk
        TZc = best_Z[zz, yy, xx]                  # current best (starts at TZ0)
        TYc = best_Y[zz, yy, xx]
        TXc = best_X[zz, yy, xx]
        PZc = PZs[zz, yy, xx]
        PYc = PYs[zz, yy, xx]
        PXc = PXs[zz, yy, xx]
        found_c = best_found[zz, yy, xx]
        dist2_c = best_dist2[zz, yy, xx]

        # scan candidates one offset at a time (O(K*M) working set ~ M)
        for oz, oy, ox in offsets:
            CZ = (TZc + oz).clamp_(0, D-1)
            CY = (TYc + oy).clamp_(0, H-1)
            CX = (TXc + ox).clamp_(0, W-1)

            clin = (CZ * (H*W) + CY * W + CX)     # (M,)
            cand_valid = dest_finite.gather(0, clin)

            # distance from safe endpoint to candidate index
            dist2 = (PZc - CZ.to(dtype))**2 + (PYc - CY.to(dtype))**2 + (PXc - CX.to(dtype))**2

            improve = cand_valid & (~found_c | (dist2 < dist2_c))
            # update only where improvement happens
            if improve.any():
                TZc = torch.where(improve, CZ, TZc)
                TYc = torch.where(improve, CY, TYc)
                TXc = torch.where(improve, CX, TXc)
                dist2_c = torch.where(improve, dist2, dist2_c)
                found_c = found_c | cand_valid

        # write back this chunk’s results
        best_Z[zz, yy, xx] = TZc
        best_Y[zz, yy, xx] = TYc
        best_X[zz, yy, xx] = TXc
        best_found[zz, yy, xx] = found_c
        best_dist2[zz, yy, xx] = dist2_c
        s = e

    # build final snapped displacement
    snapped = torch.stack([
        best_Z.to(dtype) - Z,
        best_Y.to(dtype) - Y,
        best_X.to(dtype) - X
    ], dim=0)

    # honor keep_original_if_none
    if keep_original_if_none and (~best_found).any().item():
        snapped = torch.where(best_found.unsqueeze(0), snapped, flow)

    # leave invalid sources unchanged
    snapped = torch.where(src_finite.unsqueeze(0), snapped, flow)
    return snapped



@torch.no_grad()
def point_vectors_to_centers_nanaware(
    directions: torch.Tensor,
    iters: int = 50,
    step: float = 1.0,
    mask_thresh: float = 0.99
):
    """
    Follow the flow field to a sink/center from each voxel, respecting NaN 'medium' regions.
    directions: (3,D,H,W) with order [dz,dy,dx]; NaNs mark medium/invalid.
    Returns:    (3,D,H,W) displacement-to-center; start voxels that are NaN -> NaN.
    """
    assert directions.ndim == 4 and directions.shape[0] == 3, "Expected (3,D,H,W)"
    device, dtype = directions.device, directions.dtype
    _, D, H, W = directions.shape

    start_valid = torch.isfinite(directions).all(dim=0)  # (D,H,W)

    flow = directions.clone()
    flow[~torch.isfinite(flow)] = 0
    mask = torch.isfinite(directions).all(dim=0, keepdim=True).to(directions.dtype)  # (1,D,H,W)

    flow = flow.unsqueeze(0)   # (1,3,D,H,W)
    mask = mask.unsqueeze(0)   # (1,1,D,H,W)

    z, y, x = torch.meshgrid(
        torch.arange(D, device=device, dtype=dtype),
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    base = torch.stack([z, y, x], dim=0)  # (3,D,H,W)
    pos  = base.clone()

    # Keep NaN-start voxels stationary
    pos[:, ~start_valid] = base[:, ~start_valid]

    def vox2norm(pz, py, px):
        gx = 2.0 * px / (W - 1) - 1.0
        gy = 2.0 * py / (H - 1) - 1.0
        gz = 2.0 * pz / (D - 1) - 1.0
        return torch.stack([gx, gy, gz], dim=-1)  # (D,H,W,3)

    for _ in range(iters):
        grid = vox2norm(pos[0], pos[1], pos[2]).unsqueeze(0)  # (1,D,H,W,3)
        v = F.grid_sample(flow, grid, mode='bilinear',
                          padding_mode='border', align_corners=True)[0]       # (3,D,H,W)
        w = F.grid_sample(mask, grid, mode='bilinear',
                          padding_mode='border', align_corners=True)[0, 0]    # (D,H,W)

        move_ok = (w >= mask_thresh) & start_valid
        pos[:, move_ok] = pos[:, move_ok] + step * v[:, move_ok]

        pos[0].clamp_(0, D - 1)
        pos[1].clamp_(0, H - 1)
        pos[2].clamp_(0, W - 1)

    out = pos - base  # (3,D,H,W)
    out[:, ~start_valid] = torch.nan
    return out


@torch.no_grad()
def add_inward_bias_to_directions(
    directions: torch.Tensor,
    spacing=(1.0, 1.0, 1.0),
    inward_bias: float = 0.0,
):
    """
    directions: (C,3,D,H,W) or (3,D,H,W)
      channels order: [dz, dy, dx]
    Returns a new tensor with inward-bias added where components are finite.
    """
    if inward_bias == 0.0:
        return directions

    sz, sy, sx = map(float, spacing)

    # Normalize shape to (C,3,D,H,W)
    squeeze_c = False
    if directions.ndim == 4 and directions.shape[0] == 3:
        directions = directions.unsqueeze(0)  # -> (1,3,D,H,W)
        squeeze_c = True
    assert directions.ndim == 5 and directions.shape[1] == 3, "Expected (C,3,D,H,W) or (3,D,H,W)"

    C, _, D, H, W = directions.shape
    dz = directions[:, 0].clone()
    dy = directions[:, 1].clone()
    dx = directions[:, 2].clone()

    dtype = directions.dtype
    device = directions.device

    # Finite mask per component
    fz = torch.isfinite(dz)
    fy = torch.isfinite(dy)
    fx = torch.isfinite(dx)

    # Overall validity mask (voxel is valid if all 3 comps are finite)
    m = fz & fy & fx  # (C,D,H,W)

    def shift_bool(t, dzs, dys, dxs, fill=False):
        out = torch.empty_like(t, dtype=torch.bool, device=device)
        out.fill_(fill)
        z0 = max(dzs, 0); z1 = D + min(dzs, 0); zs = slice(z0, z1)
        y0 = max(dys, 0); y1 = H + min(dys, 0); ys = slice(y0, y1)
        x0 = max(dxs, 0); x1 = W + min(dxs, 0); xs = slice(x0, x1)
        zs_src = slice(z0 - dzs, z1 - dzs)
        ys_src = slice(y0 - dys, y1 - dys)
        xs_src = slice(x0 - dxs, x1 - dxs)
        out[:, zs, ys, xs] = t[:, zs_src, ys_src, xs_src]
        return out

    # Neighbor validity masks
    mpz, mmz = shift_bool(m, -1, 0, 0, False), shift_bool(m, 1, 0, 0, False)  # +z, -z
    mpy, mmy = shift_bool(m, 0, -1, 0, False), shift_bool(m, 0, 1, 0, False)  # +y, -y
    mpx, mmx = shift_bool(m, 0, 0, -1, False), shift_bool(m, 0, 0, 1, False)  # +x, -x

    # Bias terms: +1 if only +side valid, -1 if only -side valid, 0 otherwise
    bz = inward_bias * (mpz.to(dtype) - mmz.to(dtype)) / sz
    by = inward_bias * (mpy.to(dtype) - mmy.to(dtype)) / sy
    bx = inward_bias * (mpx.to(dtype) - mmx.to(dtype)) / sx

    dz = torch.where(fz, dz + bz, dz)
    dy = torch.where(fy, dy + by, dy)
    dx = torch.where(fx, dx + bx, dx)

    out = torch.stack([dz, dy, dx], dim=1)  # (C,3,D,H,W)
    return out.squeeze(0) if squeeze_c else out


@torch.no_grad()
def snap_vectors_to_nearest_voxel(flow: torch.Tensor) -> torch.Tensor:
    """
    Snap a 3D vector field to the nearest voxel targets.

    Input:
      flow: (3, D, H, W) with components (dz, dy, dx). Each vector points from (z,y,x)
            to (z+dz, y+dy, x+dx) in continuous coords.

    Output:
      snapped: (3, D, H, W) where each vector now points exactly to the nearest voxel:
               ( round(z+dz), round(y+dy), round(x+dx) ), clamped into bounds.
    """
    assert flow.ndim == 4 and flow.shape[0] == 3, "flow must be (3,D,H,W)"
    _, D, H, W = flow.shape
    device, dtype = flow.device, flow.dtype

    # Base grid (same dtype as flow to avoid casts)
    z = torch.arange(D, device=device, dtype=dtype)
    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')  # (D,H,W)

    dz, dy, dx = flow[0], flow[1], flow[2]

    # Endpoint -> nearest voxel (round ties-to-even), then clamp to bounds
    TZ = torch.round(Z + dz).clamp_(0, D - 1)
    TY = torch.round(Y + dy).clamp_(0, H - 1)
    TX = torch.round(X + dx).clamp_(0, W - 1)

    # New displacement = snapped voxel minus origin voxel
    snapped_dz = (TZ - Z)
    snapped_dy = (TY - Y)
    snapped_dx = (TX - X)

    snapped = torch.stack([snapped_dz, snapped_dy, snapped_dx], dim=0).to(dtype)
    return snapped


@torch.no_grad()
def displacements_to_coords(out: torch.Tensor, round_to_int: bool = True):
    """
    out: (3,D,H,W) or (N,3,D,H,W) displacement field in voxel units
         (dz, dy, dx) = vector FROM each voxel TO its target.
    Returns:
      coords: same shape as out, but channels hold ABSOLUTE target coords (z,y,x).
              If round_to_int=True -> integer voxel indices (clamped in-bounds).
              Else -> float coords.
    """
    nanMask = (torch.isnan(out))
    has_batch = (out.ndim == 5)
    if not has_batch: out = out.unsqueeze(0)              # (1,3,D,H,W)
    N, C, D, H, W = out.shape; assert C == 3
    device, dtype = out.device, out.dtype

    z = torch.arange(D, device=device, dtype=dtype)
    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')       # (D,H,W)

    base = torch.stack([Z, Y, X], dim=0).unsqueeze(0)      # (1,3,D,H,W)
    coords = base + out                                    # absolute targets

    if round_to_int:
        coords = coords.round()
        coords[:,0].clamp_(0, D-1); coords[:,1].clamp_(0, H-1); coords[:,2].clamp_(0, W-1)
        coords = coords.to(torch.float32)
    if not has_batch:
        coords = coords.squeeze(0)


    coords[nanMask] = torch.nan
    return coords

@torch.no_grad()
def point_vectors_to_centers_nanaware2(
    directions: torch.Tensor,
    iters: int = 50,
    step: float = 1.0,
    mask_thresh: float = 0.99,
    repel: float = 0.0,   # inward push strength (voxels per step)
):
    """
    Follow the flow field to a sink/center from each voxel, respecting NaN 'medium' regions.
    Adds optional inward push away from NaN regions near boundaries (repel > 0).
    directions: (3,D,H,W) [dz,dy,dx]; NaNs mark medium/invalid.
    Returns:    (3,D,H,W) displacement-to-center; NaN at NaN-start voxels.
    """
    assert directions.ndim == 4 and directions.shape[0] == 3, "Expected (3,D,H,W)"
    device, dtype = directions.device, directions.dtype
    _, D, H, W = directions.shape

    start_valid = torch.isfinite(directions).all(dim=0)  # (D,H,W)

    flow = directions.clone()
    flow[~torch.isfinite(flow)] = 0
    mask0 = torch.isfinite(directions).all(dim=0, keepdim=True).to(directions.dtype)  # (1,D,H,W)

    flow = flow.unsqueeze(0)   # (1,3,D,H,W)
    mask = mask0.unsqueeze(0)  # (1,1,D,H,W)

    # --- Precompute a smooth mask and its spatial gradient for boundary normals ---
    # smooth (makes gradient well-defined near edges)
    mask_smooth = torch.nn.functional.avg_pool3d(mask, kernel_size=3, stride=1, padding=1)
    # central-diff kernels
    kx = torch.tensor([[-1.0, 0.0, 1.0]], device=device, dtype=dtype).view(1,1,1,1,3)
    ky = torch.tensor([[-1.0, 0.0, 1.0]], device=device, dtype=dtype).view(1,1,1,3,1)
    kz = torch.tensor([[-1.0, 0.0, 1.0]], device=device, dtype=dtype).view(1,1,3,1,1)
    gx = torch.nn.functional.conv3d(mask_smooth, kx, padding=(0,0,1))
    gy = torch.nn.functional.conv3d(mask_smooth, ky, padding=(0,1,0))
    gz = torch.nn.functional.conv3d(mask_smooth, kz, padding=(1,0,0))
    grad_mask = torch.cat([gz, gy, gx], dim=1)  # (1,3,D,H,W) matches [dz,dy,dx] order

    z, y, x = torch.meshgrid(
        torch.arange(D, device=device, dtype=dtype),
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    base = torch.stack([z, y, x], dim=0)  # (3,D,H,W)
    pos  = base.clone()

    # Keep NaN-start voxels stationary
    pos[:, ~start_valid] = base[:, ~start_valid]

    def vox2norm(pz, py, px):
        gx = 2.0 * px / (W - 1) - 1.0
        gy = 2.0 * py / (H - 1) - 1.0
        gz = 2.0 * pz / (D - 1) - 1.0
        return torch.stack([gx, gy, gz], dim=-1)  # (D,H,W,3)

    eps = torch.finfo(dtype).eps

    for _ in range(iters):
        grid = vox2norm(pos[0], pos[1], pos[2]).unsqueeze(0)  # (1,D,H,W,3)

        v = torch.nn.functional.grid_sample(
            flow, grid, mode='bilinear', padding_mode='border', align_corners=True
        )[0]  # (3,D,H,W)

        w = torch.nn.functional.grid_sample(
            mask, grid, mode='bilinear', padding_mode='border', align_corners=True
        )[0, 0]  # (D,H,W)

        move_ok = (w >= mask_thresh) & start_valid

        # Optional inward push away from NaNs (use gradient of mask toward interior)
        if repel > 0.0:
            g = torch.nn.functional.grid_sample(
                grad_mask, grid, mode='bilinear', padding_mode='border', align_corners=True
            )[0]  # (3,D,H,W)
            g_norm = g.norm(dim=0).clamp_min(eps)
            b = repel * (g / g_norm)  # push toward increasing mask (interior)
        else:
            b = 0.0

        # Only advance where valid enough; bias only applied where we advance
        if isinstance(b, float):
            pos[:, move_ok] = pos[:, move_ok] + step * v[:, move_ok]
        else:
            pos[:, move_ok] = pos[:, move_ok] + step * (v[:, move_ok] + b[:, move_ok])

        # Clamp to bounds
        pos[0].clamp_(0, D - 1)
        pos[1].clamp_(0, H - 1)
        pos[2].clamp_(0, W - 1)

    out = pos - base  # (3,D,H,W)
    out[:, ~start_valid] = torch.nan
    return out


def modelFlowToCenter(gen_outputs: torch.tensor):
    tmp = gen_outputs[0,:3,:].clone()
    logits = gen_outputs[0, 3:, ...].clone()          # [3, 200, 200, 200]
    tmpidx = logits.argmax(dim=0, keepdim=True) # [1, 200, 200, 200]
    tmp_one_hot = torch.zeros_like(logits).scatter_(0, tmpidx, 1)
    tmpMask = (tmp_one_hot[2] == 1)
    tmpMask = tmpMask.repeat(3, 1, 1, 1)
    tmp[tmpMask] = torch.nan
    i=0
    mask = None
    ib = 0.08
    previous_tmp = None
    prevLast = tmp.shape[1]*tmp.shape[2]*tmp.shape[3]
    last = prevLast
    while previous_tmp is None or last<prevLast or last > 1000:
        previous_tmp = tmp.clone()
        prevLast=last
        # Save current state
        print(str(i))


        # Step 1: always sum
        tmp, mask = sum_with_next_from(tmp, tmp, neighbor_vals=tmp, avoid_self=True, mask=mask)


        # Step 2: either snap or bias
        if (i + 1) % 10 == 0:
            tmp = snap_vectors_to_nearest_non_nanV3(tmp, search_radius=2, max_chunk_voxels=800_000)
        elif i==3:
            tmp = point_vectors_to_centers_nanaware(tmp, iters=100, step=1.0, mask_thresh=0.9)
        else:
            tmp = add_inward_bias_to_directions(tmp, inward_bias=ib)

        i += 1

        # Optional: break if it’s spiraling into the tensor abyss
        if i > 100:
            print("Max iterations reached.")
            break
        if i>30:
            last = (~(torch.isclose(previous_tmp, tmp, atol=1, equal_nan=True) | torch.isclose(previous_tmp, tmp, rtol=10, equal_nan=True))).float().sum().item()
            print("Last: ", last)
        else:
            last -= 1

    tmp = snap_vectors_to_nearest_voxel(tmp)
    tmp, mask = sum_with_next_from(tmp, tmp, neighbor_vals=tmp, avoid_self=True, mask=mask)
    return tmp


def modelFlowToCenter2(gen_outputs: torch.tensor, iters=100, step=1, mask_thresh=0, repel=0.2):
    tmp = gen_outputs[0,:3,:].clone()
    logits = gen_outputs[0, 3:, ...].clone()          # [3, 200, 200, 200]
    tmpidx = logits.argmax(dim=0, keepdim=True) # [1, 200, 200, 200]
    tmp_one_hot = torch.zeros_like(logits).scatter_(0, tmpidx, 1)
    tmpMask = (tmp_one_hot[2] == 1)
    tmpMask = tmpMask.repeat(3, 1, 1, 1)
    tmp[tmpMask] = torch.nan
    mask=None
    tmp = point_vectors_to_centers_nanaware2(tmp, iters=iters, step=step, mask_thresh=mask_thresh, repel=repel)
    return tmp





def quiver_slice_zyx(
    distTensor,
    axis='z', index=0, mode='disp',
    color_by='mag',          # 'body', 'mag', 'angle', or 'none'
    cmap='tab20',
    stride=1,
    exclude_boundary_target=False, exclude_radius=1.0,
    arrowScale=1,
    savePath=""              # <---- new arg
):
    """
    Plot a dense 2D quiver slice from a 3- or 4-channel tensor.
    If savePath != "": save image to that path instead of opening a window.
    """
    # ---- to numpy float32 ----
    try:
        import torch, numpy as np, matplotlib.pyplot as plt
        if isinstance(distTensor, torch.Tensor):
            A = distTensor.detach().float().cpu().numpy().copy()
        else:
            import numpy as np, matplotlib.pyplot as plt
            A = np.asarray(distTensor, dtype=np.float32)
    except Exception:
        import numpy as np, matplotlib.pyplot as plt
        A = np.asarray(distTensor, dtype=np.float32)

    assert A.ndim == 4 and A.shape[0] in (3,4), "dist must be (3,Z,Y,X) or (4,Z,Y,X)"

    if A.shape[0] == 4:
        uz, uy, ux, r = A[0], A[1], A[2], A[3]
        valid_full = (r >= 0.0)
    else:
        uz, uy, ux = A[0], A[1], A[2]
        r = np.ones_like(uz, dtype=np.float32)
        valid_full = np.isfinite(uz) & np.isfinite(uy) & np.isfinite(ux)

    Z, Y, X = uz.shape
    import numpy as np, matplotlib.pyplot as plt
    if axis == 'z':
        k = int(np.clip(index, 0, Z-1))
        UX, UY, UZ = ux[k], uy[k], uz[k]
        R = r[k]
        U_in, V_in = UX, UY
        gx, gy = np.meshgrid(np.arange(X, dtype=np.float32),
                             np.arange(Y, dtype=np.float32))
        w, h = X, Y
        valid = valid_full[k]
        xlabel, ylabel = "X", "Y"
    elif axis == 'y':
        k = int(np.clip(index, 0, Y-1))
        UX, UY, UZ = ux[:, k, :], uy[:, k, :], uz[:, k, :]
        R = r[:, k, :]
        U_in, V_in = UX, UZ
        gx, gy = np.meshgrid(np.arange(X, dtype=np.float32),
                             np.arange(Z, dtype=np.float32))
        w, h = X, Z
        valid = valid_full[:, k, :]
        xlabel, ylabel = "X", "Z"
    elif axis == 'x':
        k = int(np.clip(index, 0, X-1))
        UX, UY, UZ = ux[:, :, k], uy[:, :, k], uz[:, :, k]
        R = r[:, :, k]
        U_in, V_in = UY, UZ
        gx, gy = np.meshgrid(np.arange(Y, dtype=np.float32),
                             np.arange(Z, dtype=np.float32))
        w, h = Y, Z
        valid = valid_full[:, :, k]
        xlabel, ylabel = "Y", "Z"
    else:
        raise ValueError("axis must be 'x','y','z'")

    # ---- in-plane vectors ----
    if A.shape[0] == 4:
        if mode == 'disp':
            U, V = U_in * R, V_in * R
        elif mode == 'unit':
            U, V = U_in, V_in
        else:
            raise ValueError("mode must be 'disp' or 'unit'")
    else:
        if mode == 'disp':
            U, V = U_in, V_in
        elif mode == 'unit':
            mag2d = np.hypot(U_in, V_in) + 1e-12
            U, V = U_in / mag2d, V_in / mag2d
        else:
            raise ValueError("mode must be 'disp' or 'unit'")

    iy, ix = np.nonzero(valid)
    if stride and stride > 1 and iy.size:
        keep = ((iy % stride) == 0) & ((ix % stride) == 0)
        iy, ix = iy[keep], ix[keep]

    if iy.size and exclude_boundary_target:
        tip_x_all = gx[iy, ix] + U_in[iy, ix] * (R[iy, ix] if A.shape[0]==4 and mode=='disp' else 1.0)
        tip_y_all = gy[iy, ix] + V_in[iy, ix] * (R[iy, ix] if A.shape[0]==4 and mode=='disp' else 1.0)
        dist_left   = tip_x_all
        dist_right  = (w - 1) - tip_x_all
        dist_top    = tip_y_all
        dist_bottom = (h - 1) - tip_y_all
        d_border = np.minimum(np.minimum(dist_left, dist_right),
                              np.minimum(dist_top,  dist_bottom))
        mid_x = (w - 1) * 0.5
        tie1 = -tip_y_all
        tie2 = np.abs(tip_x_all - mid_x)
        order = np.lexsort((tie2, tie1, d_border))
        i_sel = order[0]
        bx, by = tip_x_all[i_sel], tip_y_all[i_sel]
        keep2 = ((tip_x_all - bx)**2 + (tip_y_all - by)**2) > (exclude_radius**2)
        iy, ix = iy[keep2], ix[keep2]

    Xp = gx[iy, ix].astype(np.float32, copy=False)
    Yp = gy[iy, ix].astype(np.float32, copy=False)
    Up = U[iy, ix].astype(np.float32, copy=False)
    Vp = V[iy, ix].astype(np.float32, copy=False)

    C = None
    use_cmap = False
    if color_by == 'body':
        tipx = (gx[iy, ix] + Up).round().astype(int) if (A.shape[0]==3 or mode!='disp') \
               else (gx[iy, ix] + (U_in[iy, ix]*R[iy, ix])).round().astype(int)
        tipy = (gy[iy, ix] + Vp).round().astype(int) if (A.shape[0]==3 or mode!='disp') \
               else (gy[iy, ix] + (V_in[iy, ix]*R[iy, ix])).round().astype(int)
        tipx = np.clip(tipx, 0, w-1); tipy = np.clip(tipy, 0, h-1)
        labels = tipy * w + tipx
        _, C = np.unique(labels, return_inverse=True)
        C = C.astype(np.float32); use_cmap = True
    elif color_by == 'mag':
        C = np.hypot(Up, Vp).astype(np.float32); use_cmap = True
    elif color_by == 'angle':
        C = ((np.arctan2(Vp, Up) + np.pi) / (2*np.pi)).astype(np.float32); use_cmap = True
    elif color_by == 'none':
        pass
    else:
        raise ValueError("color_by must be 'body', 'mag', 'angle', or 'none'")

    plt.figure(figsize=(12, 12))
    if Xp.size:
        if use_cmap:
            plt.quiver(Xp, Yp, Up, Vp, C, cmap=cmap, angles='xy',
                       scale_units='xy', scale=arrowScale, pivot='tail')
        else:
            plt.quiver(Xp, Yp, Up, Vp, angles='xy',
                       scale_units='xy', scale=arrowScale, pivot='tail')

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    plt.tight_layout()

    if savePath != "":
        plt.savefig(savePath, dpi=300)
        plt.close()
    else:
        plt.show()

# --- CHUNK 1: helpers (parallel parse + parallel slab fill) ---
import os, mmap, numpy as np
from math import ceil
from concurrent.futures import ProcessPoolExecutor

def _compute_line_ranges(path, workers, target_chunk_bytes=64 << 20):
    size = os.path.getsize(path)
    if size == 0:
        return [(0, 0)]
    with open(path, 'rb') as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        n = min(workers, max(1, ceil(size / target_chunk_bytes)))
        starts = [0]
        for i in range(1, n):
            pos = (size * i) // n
            nl = mm.find(b'\n', pos)
            if nl == -1:
                break
            starts.append(nl + 1)
        starts.append(size)
    return [(starts[i], starts[i+1]) for i in range(len(starts)-1)]

def _parse_chunk_worker(args):
    path, start, end = args
    rows = []
    wall_flags = []
    max_x = max_y = max_z = -1
    with open(path, 'rb') as f:
        f.seek(start)
        data = f.read(end - start)
    for line in data.splitlines():
        parts = line.split()
        if len(parts) != 8:
            continue
        # parts: [cell_ID, cell_type, x1, x2, y1, y2, z1, z2] (bytes)
        cid = int(parts[0])
        x1 = int(parts[2]); x2 = int(parts[3])
        y1 = int(parts[4]); y2 = int(parts[5])
        z1 = int(parts[6]); z2 = int(parts[7])
        rows.append((cid, x1, x2, y1, y2, z1, z2))
        wall_flags.append(1 if parts[1] == b"Wall" else 0)
        if x2 > max_x: max_x = x2
        if y2 > max_y: max_y = y2
        if z2 > max_z: max_z = z2
    if rows:
        rows = np.asarray(rows, dtype=np.int32)
        wall_flags = np.asarray(wall_flags, dtype=np.uint8)
    else:
        rows = np.empty((0,7), dtype=np.int32)
        wall_flags = np.empty((0,), dtype=np.uint8)
    return rows, wall_flags, max_x, max_y, max_z

def parse_boxes(path, workers=None, target_chunk_bytes=64 << 20, parallel=False):
    """
    Parse axis-aligned boxes from `path` and build a labeled volume.
    Returns (vol, wallID, max_x, max_y, max_z) where:
      - vol: (1, Z, Y, X) float32 with -1 background and cell IDs written
      - wallID: last 'Wall' cell_ID encountered in file order (or -1 if none)
      - max_x, max_y, max_z: extents inferred from boxes
    """
    if (not parallel):
        voxels = []
        max_x = max_y = max_z = -1
        append_voxel = voxels.append
        _to_int = int

        with open(path, 'r', buffering=target_chunk_bytes) as f:
            for line in f:
                parts = line.split()
                if len(parts) != 8:
                    continue
                cell_ID, cell_type = parts[0], parts[1]
                x1 = _to_int(parts[2]); x2 = _to_int(parts[3])
                y1 = _to_int(parts[4]); y2 = _to_int(parts[5])
                z1 = _to_int(parts[6]); z2 = _to_int(parts[7])
                append_voxel((cell_ID, cell_type, x1, x2, y1, y2, z1, z2))
                if x2 > max_x: max_x = x2
                if y2 > max_y: max_y = y2
                if z2 > max_z: max_z = z2

        # allocate once as int32 (faster writes), cast once at end to match float32
        vol = np.full((1, max_z + 1, max_y + 1, max_x + 1), -1, dtype=np.int32)
        wallID = -1

        for cell_ID, cell_type, x1, x2, y1, y2, z1, z2 in voxels:
            cid = int(cell_ID)
            vol[0, z1 : z2 + 1, y1 : y2 + 1, x1 : x2 + 1] = cid
            if cell_type == "Wall":
                wallID = cid

        vol = vol.astype(np.float32, copy=False)
        return vol, wallID, max_x, max_y, max_z

    # ---------- parallel path ----------
    workers = workers or os.cpu_count() or 4
    ranges = _compute_line_ranges(path, workers, target_chunk_bytes)

    rows_list, flags_list = [], []
    max_x = max_y = max_z = -1

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for rows, flags, mx, my, mz in ex.map(
            _parse_chunk_worker, [(path, a, b) for a, b in ranges], chunksize=1
        ):
            rows_list.append(rows)
            flags_list.append(flags)
            if mx > max_x: max_x = mx
            if my > max_y: max_y = my
            if mz > max_z: max_z = mz

    rows = np.concatenate(rows_list, axis=0) if rows_list else np.empty((0, 7), np.int32)
    flags = np.concatenate(flags_list, axis=0) if flags_list else np.empty((0,), np.uint8)

    # last Wall ID by original order (chunks concatenated in file order)
    wallID = -1
    if flags.size:
        idx = np.flatnonzero(flags)
        if idx.size:
            wallID = int(rows[idx[-1], 0])

    # build the volume like the non-parallel path (via slab filler)
    Z, Y, X = max_z + 1, max_y + 1, max_x + 1
    vol = np.full((1, Z, Y, X), -1, dtype=np.int32)
    vol = fill_slabs(vol, rows, max_x, max_y, max_z, parallel=True)
    vol = vol.astype(np.float32, copy=False)
    return vol, wallID, max_x, max_y, max_z


def _fill_slab_worker(args):
    z0, z1, X, Y, rows = args  # rows: (M,7) -> [id,x1,x2,y1,y2,z1,z2]
    sub = np.full((z1 - z0, Y, X), -1, dtype=np.int32)
    # preserve input order within slab (last-write-wins locally)
    for i in range(rows.shape[0]):
        cid, x1, x2, y1, y2, z1r, z2r = rows[i]
        zs = z1r if z1r > z0 else z0
        ze = z2r if z2r < (z1 - 1) else (z1 - 1)
        if zs > ze:
            continue
        for z in range(zs, ze + 1):
            plane = sub[z - z0]
            for y in range(y1, y2 + 1):
                plane[y, x1:x2 + 1] = cid
    return z0, sub

def fill_slabs(vol, rows, max_x, max_y, max_z, slab=16, workers=None, parallel=False):
    Z, Y, X = max_z + 1, max_y + 1, max_x + 1
    assert vol.shape == (1, Z, Y, X) and vol.dtype == np.int32

    if (not parallel):
        # Non-parallel path: process slabs sequentially
        for z0 in range(0, Z, slab):
            z1 = min(z0 + slab, Z)
            sel = (rows[:, 5] <= (z1 - 1)) & (rows[:, 6] >= z0)
            rows_slab = rows[sel]
            _, sub = _fill_slab_worker((z0, z1, X, Y, rows_slab))
            vol[0, z0:z0 + sub.shape[0]] = sub
        return vol

    workers = workers or os.cpu_count() or 4
    # build per-slab jobs with rows clipped by z-range
    jobs = []
    for z0 in range(0, Z, slab):
        z1 = min(z0 + slab, Z)
        sel = (rows[:, 5] <= (z1 - 1)) & (rows[:, 6] >= z0)
        rows_slab = rows[sel]
        jobs.append((z0, z1, X, Y, rows_slab))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for z0, sub in ex.map(_fill_slab_worker, jobs, chunksize=1):
            vol[0, z0:z0 + sub.shape[0]] = sub
    return vol




def condense_single_channel(x: torch.Tensor, channel_dim: int = 0) -> torch.Tensor:
    """
    Collapse the channels of `x` into one channel using these rules:
      - If ALL channels at a voxel are +/−inf -> output inf at that voxel.
      - If EXACTLY ONE channel is finite -> take that finite value.
      - If MORE THAN ONE channel is finite -> raise ValueError.

    Args:
        x: tensor with a channel dimension (e.g., (C,D,H,W) or (N,C,H,W); you choose which is channel_dim)
        channel_dim: index of the channel dimension in `x` (default 0)

    Returns:
        Tensor with the channel dimension removed.

    Raises:
        ValueError if any voxel has >1 finite values across channels.
    """
    # move channel axis to front for simpler logic
    t = x.movedim(channel_dim, 0)  # (C, ...)
    # treat NaNs as "not a value" too (only infinities matter for the rule)
    finite = ~torch.isinf(t) & ~torch.isnan(t)

    # count finite entries per voxel
    count = finite.sum(dim=0)

    # error if more than one finite value exists at any voxel
    conflict = count > 1
    if conflict.any():
        first = torch.nonzero(conflict, as_tuple=False)[0]
        idx_tuple = tuple(int(i) for i in first.tolist())
        raise ValueError(f"More than one finite value at position {idx_tuple} across channels.")

    # for voxels with exactly one finite value, pick its channel index
    # (argmax chooses first True; safe because count<=1 at all kept voxels)
    pick_idx = torch.argmax(finite.to(torch.int64), dim=0)  # (...,)

    # gather the chosen values (one per voxel)
    gathered = t.gather(0, pick_idx.unsqueeze(0)).squeeze(0)  # same shape as per-voxel (...,)

    # initialize output as +inf; fill only where exactly one finite value exists
    out = torch.full_like(gathered, float("nan"))
    sel = (count == 1)
    out[sel] = gathered[sel]
    return out



import torch

@torch.no_grad()
def zero_components_toward_adjacent_nans(flow: torch.Tensor) -> torch.Tensor:
    """
    flow: (C, 3, D, H, W) in [dz, dy, dx] order.
    Zero a component if it points toward an adjacent NaN voxel; out-of-bounds counts as NaN.
    """
    assert flow.ndim == 5 and flow.shape[1] == 3, "Expected (C,3,D,H,W)"
    C, _, D, H, W = flow.shape

    out = flow.clone()
    valid = torch.isfinite(out).all(dim=1)  # (C,D,H,W)

    dz = out[:, 0, ...]
    dy = out[:, 1, ...]
    dx = out[:, 2, ...]

    # Neighbor-invalid (True) masks, treating OOB as invalid
    cur_has_left_nan  = torch.ones_like(valid, dtype=torch.bool)
    cur_has_right_nan = torch.ones_like(valid, dtype=torch.bool)
    cur_has_up_nan    = torch.ones_like(valid, dtype=torch.bool)
    cur_has_down_nan  = torch.ones_like(valid, dtype=torch.bool)
    cur_has_front_nan = torch.ones_like(valid, dtype=torch.bool)
    cur_has_back_nan  = torch.ones_like(valid, dtype=torch.bool)

    # x-1 (left): interior depends on neighbor validity
    cur_has_left_nan[..., :, :, 1:]  = ~valid[..., :, :, :-1]
    # x+1 (right)
    cur_has_right_nan[..., :, :, :-1] = ~valid[..., :, :, 1:]
    # y-1 (up)
    cur_has_up_nan[..., :, 1:, :]   = ~valid[..., :, :-1, :]
    # y+1 (down)
    cur_has_down_nan[..., :, :-1, :] = ~valid[..., :, 1:, :]
    # z-1 (front)
    cur_has_front_nan[..., 1:, :, :]  = ~valid[..., :-1, :, :]
    # z+1 (back)
    cur_has_back_nan[..., :-1, :, :] = ~valid[..., 1:, :, :]

    # Only modify sites that are valid themselves
    cur_valid = valid

    mask_dx_zero = ((dx < 0) & cur_has_left_nan)  | ((dx > 0) & cur_has_right_nan)
    mask_dy_zero = ((dy < 0) & cur_has_up_nan)    | ((dy > 0) & cur_has_down_nan)
    mask_dz_zero = ((dz < 0) & cur_has_front_nan) | ((dz > 0) & cur_has_back_nan)

    mask_dx_zero &= cur_valid
    mask_dy_zero &= cur_valid
    mask_dz_zero &= cur_valid

    dx[mask_dx_zero] = 0
    dy[mask_dy_zero] = 0
    dz[mask_dz_zero] = 0

    out[:, 0, ...] = dz
    out[:, 1, ...] = dy
    out[:, 2, ...] = dx
    return out




def parse_voxel_file_for_distance(path, device=torch.device("cpu"), loadExisting=True, saveLoaded=True, parallel=False):
    """
    Read a text-based voxel file where each line specifies a solid block:
      ID Type x1 x2 y1 y2 z1 z2
    Build and return a one-hot (1, D, H, W) volume as a torch tensor.
    """
    import os
    voxels = []
    max_x = max_y = max_z = 0

    if loadExisting and os.path.isfile(os.path.join(str(path).removesuffix(".piff") + "quickload.pt")):
        try:
            tensor = torch.load(os.path.join(str(path).removesuffix(".piff") + "quickload.pt"), weights_only=True, map_location=device)
            centers = tensor["centers"]
            flow = tensor["flow"]
            bodyMask = tensor["bodyMask"]
            wallMask = tensor["wallMask"]
            mediumMask = tensor["mediumMask"]
        except RuntimeError as e:
            if "PytorchStreamReader failed reading zip archive" in str(e):
                print(f"[!] Corrupted file detected: {path}, deleting and rebuilding...")
                try:
                    os.remove(os.path.join(str(path).removesuffix(".piff") + "quickload.pt"))
                except Exception as rm_err:
                    print(f"   Failed to delete {os.path.join(str(path).removesuffix(".piff") + "quickload.pt")}: {rm_err}")
                    return parse_voxel_file_for_distance(path, loadExisting, saveLoaded)
                return parse_voxel_file_for_distance(path, loadExisting, saveLoaded)
        return torch.cat([flow, bodyMask, wallMask,mediumMask], dim=0), centers.squeeze(0) #, centers, wallID, wallMask




#########################################


    import numpy as np, os


    vol, wallID, max_x, max_y, max_z = parse_boxes(path, workers=8, target_chunk_bytes=64 << 20, parallel=parallel)

    #vol = np.full((1, max_z + 1, max_y + 1, max_x + 1), -1, dtype=np.int32)


    #fill_slabs(vol, rows, max_x, max_y, max_z, slab=16, workers=8, parallel=parallel)
    
    #vol = vol.astype(np.float32, copy=False)


#########################################


    vol = pad_crop_resize(vol)

    wallMask = torch.from_numpy(vol == wallID)
    mediumMask = torch.from_numpy(vol == -1)
    bodyMask = ((~wallMask) & (~mediumMask))


    centers = edt_maxima_per_id(vol[0], workers=8, parallel=parallel)

    #centers = edt_maxima_per_id2()

    #coords = torch.zeros(3, 200, 200, 200,device=device)
    #coords = torch.full_like(coords, -1)

    #tensor = torch.from_numpy(vol)
    #tensor = tensor.to(device)

    #for id_val, z, y, x in centers:
        #mask = (tensor[0] == id_val)
        #coords[0][mask] = z
        #coords[1][mask] = y
        #coords[2][mask] = x

    #mask = (coords[0] == -1) & (coords[1] == -1) & (coords[2] == -1)
    #dist = unit_dir_and_distance_zyx(coords)


    #dist = dist.unsqueeze(0)
    vol = torch.from_numpy(vol)
    vol = vol.to(device)
    wallMask = wallMask.to(device)
    bodyMask = bodyMask.to(device)
    mediumMask = mediumMask.to(device)
    centers = torch.tensor(centers, dtype=torch.float16, device=device)

    OneChannelDistances3,MultiChannelDistances3 = oneHotToDistance_fast(vol, centers)

    vol=None # RELEASE RESOURCES
    tensor = None

    smoothedDistance = smoothDistance(MultiChannelDistances3)

    flow,tmp2 = masked_gradient3d(smoothedDistance, inward_bias=0)

    flow = flow*-1

    flow = zero_components_toward_adjacent_nans(flow)

    flow = condense_single_channel(flow)

    #flow = flow * -1
    
    flow[(torch.isnan(flow))] = 0

    # x: [N, C, ...]; last 3 channels are the masks
    m = mediumMask.float() + bodyMask.float() + wallMask.float()       # take the last 3 channels

    # per-voxel sum across the 3 mask channels
    sumc = m.sum(dim=0)              # shape: [N, ...]

    # violations
    too_many = sumc > 1              # more than one '1'
    none     = sumc < 1              # zero '1's
    ok       = (sumc == 1)     # everything exactly one-hot

    #print("OK one-hot everywhere:", ok.sum().item())
    #print("voxels with >1:", int(too_many.sum().item()))
    #print("voxels with 0:",  int(none.sum().item()))

    if (none.any() or too_many.any() or torch.isnan(flow).any()):
        print("Bad")

    assert not none.any(),     f"{int(none.sum().item())} voxels have 0 of the 3 masks set"
    assert not too_many.any(), f"{int(too_many.sum().item())} voxels have >1 of the 3 masks set"

    mask3 = mediumMask.bool().expand_as(flow)   # [3,D,H,W]
    flow[mask3] = 0 # zeros where mediumMask is True

    assert not torch.isnan(flow).any()

    if saveLoaded:
        torch.save(
        {
            "centers": centers,
            "flow": flow,
            "bodyMask": bodyMask,
            "wallMask": wallMask,
            "mediumMask": mediumMask
        },
        os.path.join(str(path).removesuffix(".piff") + "quickLoad.pt")
        )

    return torch.cat([flow, bodyMask, wallMask,mediumMask], dim=0), centers.squeeze(0) #, centers, wallID, wallMask

def drop_nearby_by_count(result: torch.Tensor, radius: float = 3.0, metric: str = "euclidean", minCount = 0):
    """
    result: [M, 4] = [count, z, y, x] (dtype can be float or int)
    radius: suppress anything within this distance of a kept point
    metric: "euclidean" | "chebyshev" | "manhattan"
    returns: filtered result with nearby lower-count rows removed
    """
    counts = result[:, 0]
    coords = result[:, 1:].to(torch.float32)  # [M,3]

    order = torch.argsort(counts, descending=True)  # process high→low
    keep = torch.zeros(result.size(0), dtype=torch.bool, device=result.device)
    suppressed = torch.zeros_like(keep)

    for idx in order:
        if suppressed[idx] or counts[idx] < minCount:
            continue
        # keep this one
        keep[idx] = True

        # compute distances to all (broadcast), then suppress neighbors
        diffs = (coords - coords[idx]).abs()  # [M,3]

        if metric == "euclidean":
            dists = torch.sqrt((diffs ** 2).sum(dim=1))       # L2
            neighbors = dists <= radius
        elif metric == "chebyshev":
            dmax = diffs.max(dim=1).values                    # L∞ (grid-friendly)
            neighbors = dmax <= radius
        elif metric == "manhattan":
            d1 = diffs.sum(dim=1)                             # L1
            neighbors = d1 <= radius
        else:
            raise ValueError("metric must be euclidean | chebyshev | manhattan")

        suppressed |= neighbors  # mark all neighbors (incl. self)
        # ensure the one we just kept stays marked as kept
        # (keep mask already set; suppressed being True here is fine)

    return result[keep]


import torch

def cluster_ids_from_coords(coords: torch.Tensor) -> torch.Tensor:
    """
    Assign a deterministic integer ID to each voxel based on its target coordinate.
    Voxels pointing to (0,0,0) OR with any NaN component get ID 0 (background).

    coords: (3, D, H, W) absolute targets (z,y,x) per voxel (floats ok, may contain NaNs).
    returns: ids (D, H, W) int64, where same (z,y,x) → same id, background → 0
    """
    assert coords.ndim == 4 and coords.shape[0] == 3, "coords must be (3,D,H,W)"
    zf, yf, xf = coords[0], coords[1], coords[2]

    # Background by NaN
    nan_bg = torch.isnan(zf) | torch.isnan(yf) | torch.isnan(xf)

    # Round only where finite; elsewhere use 0 placeholder (will be masked to bg)
    z = torch.where(nan_bg, torch.zeros_like(zf), torch.round(zf)).to(torch.int64)
    y = torch.where(nan_bg, torch.zeros_like(yf), torch.round(yf)).to(torch.int64)
    x = torch.where(nan_bg, torch.zeros_like(xf), torch.round(xf)).to(torch.int64)

    # Background by (0,0,0)
    zero_bg = (z == 0) & (y == 0) & (x == 0)
    bg = nan_bg | zero_bg

    # Deterministic 64-bit hash for (z,y,x); +1 to separate from background id 0
    id64 = ((z + 1) * 73856093) ^ ((y + 1) * 19349663) ^ ((x + 1) * 83492791)
    id64 = id64.to(torch.int64)
    id64[bg] = 0
    return id64


import torch

@torch.no_grad()
def render_cluster_slice(ids: torch.Tensor,
                         axis: str = 'z',
                         index: int = 0,
                         background: str = 'black') -> torch.Tensor:
    """
    Create an RGB image for a single 2D slice where all voxels pointing to the same
    (z,y,x) target share the same color. ID 0 (background) is black/white.

    ids:   (D,H,W) int64, typically from cluster_ids_from_coords()
    axis:  'z' | 'y' | 'x'  (which plane to render)
    index: slice index along that axis
    background: 'black' or 'white'

    returns: RGB uint8 tensor of shape (H,W,3) for 'z' slices,
             (D,W,3) for 'y' slices, or (D,H,3) for 'x' slices.
    """
    assert ids.ndim == 3, "ids must be (D,H,W)"
    D, H, W = ids.shape

    if axis == 'z':
        assert 0 <= index < D
        id2d = ids[index]             # (H,W)
    elif axis == 'y':
        assert 0 <= index < H
        id2d = ids[:, index, :]       # (D,W)
    elif axis == 'x':
        assert 0 <= index < W
        id2d = ids[:, :, index]       # (D,H)
    else:
        raise ValueError("axis must be 'z', 'y', or 'x'")

    id2d = id2d.contiguous()

    # Background mask
    bg = (id2d == 0)

    # Deterministic color hashing using int64 + modulo (avoid uint64 bitwise ops)
    u = id2d.to(torch.int64)
    r = torch.remainder(u * 0x12FAD5C3B, 256).to(torch.uint8)
    g = torch.remainder(u * 0x9E3779B97, 256).to(torch.uint8)
    b = torch.remainder(u * 0x6C8E9CF57, 256).to(torch.uint8)

    rgb = torch.stack([r, g, b], dim=-1)  # (...,3)

    # Set background to chosen color
    if background == 'white':
        bg_color = torch.tensor([255, 255, 255], dtype=torch.uint8, device=rgb.device)
    else:
        bg_color = torch.tensor([0, 0, 0], dtype=torch.uint8, device=rgb.device)

    rgb[bg] = bg_color
    return rgb


import torch

@torch.no_grad()
def snap_coords_fast(
    coords: torch.Tensor,
    centers: torch.Tensor,
    r_snap: int = 1,          # see 'interpret'
    r_neighbor: int = 2,      # see 'interpret'
    treat_zero_as_bg: bool = True,
    interpret: str = "radius" # "radius" => ±r ; "size" => exact cube of size
):
    """
    Fast snapping with NaN handling and selectable offset interpretation.

    interpret = "radius":
        r_snap=3      -> offsets in {-3..+3} (7x7x7)
        r_neighbor=2  -> offsets in {-2..+2} (5x5x5)
        NOTE: radius=0 => 1x1x1

    interpret = "size":
        r_snap=3      -> offsets in {-1,0,1} (3x3x3), centered
        r_neighbor=2  -> offsets in {0,1}   (2x2x2), anchored non-negative

    NaNs are background; (0,0,0) is background when treat_zero_as_bg=True.
    """
    assert coords.ndim == 4 and coords.shape[0] == 3, "coords must be (3,D,H,W)"
    device, dtype = coords.device, coords.dtype
    _, D, H, W = coords.shape
    out = coords.clone()

    if centers.numel() == 0:
        return out

    # --- Build center index grid (highest vote per cell) ---
    centers_f = centers[:, 1:4].to(device=device, dtype=dtype)
    centers_i = centers_f.round().to(torch.int64)
    cz = centers_i[:, 0].clamp(0, D - 1)
    cy = centers_i[:, 1].clamp(0, H - 1)
    cx = centers_i[:, 2].clamp(0, W - 1)
    votes = centers[:, 0].to(device=device, dtype=dtype)

    total = D * H * W
    lin = (cz * (H * W) + cy * W + cx).to(torch.long)
    max_votes = torch.full((total,), float("-inf"), device=device, dtype=dtype)
    max_votes.scatter_reduce_(0, lin, votes, reduce="amax", include_self=True)
    keep = votes >= max_votes.gather(0, lin)

    idx_grid = torch.full((total,), -1, device=device, dtype=torch.int32)
    kept_lin = lin[keep]
    kept_idx = torch.arange(centers_f.shape[0], device=device, dtype=torch.int32)[keep]
    idx_grid.scatter_(0, kept_lin, kept_idx)
    idx_grid = idx_grid.view(D, H, W)

    # --- Masks & predicted integer coords (safe with NaNs) ---
    has_nan = torch.isnan(out).any(dim=0)
    if treat_zero_as_bg:
        is_zero = (out[0] == 0) & (out[1] == 0) & (out[2] == 0)
        invalid_anchor = has_nan | is_zero
    else:
        invalid_anchor = has_nan

    o0 = torch.nan_to_num(out[0], nan=0.0)
    o1 = torch.nan_to_num(out[1], nan=0.0)
    o2 = torch.nan_to_num(out[2], nan=0.0)

    pr = torch.empty_like(out, dtype=torch.int64)
    pr[0] = torch.clamp(torch.round(o0), 0, D - 1).to(torch.int64)
    pr[1] = torch.clamp(torch.round(o1), 0, H - 1).to(torch.int64)
    pr[2] = torch.clamp(torch.round(o2), 0, W - 1).to(torch.int64)

    already_ok = idx_grid[pr[0], pr[1], pr[2]] >= 0
    chosen = torch.full((D, H, W), -1, device=device, dtype=torch.int32)

    # --- Offset generators ---
    def offsets_radius(r: int):
        items = []
        for dz in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    dist_inf = max(abs(dz), abs(dy), abs(dx))
                    dist_l1  = abs(dz) + abs(dy) + abs(dx)
                    items.append((dist_inf, dist_l1, dz, dy, dx))
        items.sort(key=lambda t: (t[0], t[1]))
        return [(dz, dy, dx) for _, _, dz, dy, dx in items]

    def offsets_size(size: int):
        # centered for odd sizes; non-negative (0..s-1) for even sizes (anchored)
        assert size >= 1
        if size % 2 == 1:
            r = size // 2
            return offsets_radius(r)
        else:
            items = []
            for dz in range(0, size):
                for dy in range(0, size):
                    for dx in range(0, size):
                        dist_inf = max(dz, dy, dx)
                        dist_l1  = dz + dy + dx
                        items.append((dist_inf, dist_l1, dz, dy, dx))
            items.sort(key=lambda t: (t[0], t[1]))
            return [(dz, dy, dx) for _, _, dz, dy, dx in items]

    if interpret == "radius":
        snap_offsets = offsets_radius(r_snap)
        nb_offsets   = offsets_radius(r_neighbor)
    elif interpret == "size":
        snap_offsets = offsets_size(r_snap)
        nb_offsets   = offsets_size(r_neighbor)
    else:
        raise ValueError("interpret must be 'radius' or 'size'")

    # --- Step 1: snap around predicted coord ---
    elig = (~invalid_anchor) & (~already_ok)
    pending = elig.clone()
    for dz, dy, dx in snap_offsets:
        if not pending.any():
            break
        zz = (pr[0] + dz).clamp(0, D - 1)
        yy = (pr[1] + dy).clamp(0, H - 1)
        xx = (pr[2] + dx).clamp(0, W - 1)
        cand = idx_grid[zz, yy, xx]
        hit = (cand >= 0) & pending
        if hit.any():
            chosen[hit] = cand[hit]
            pending[hit] = False

    snapped = chosen >= 0
    if snapped.any():
        csel = centers_f[chosen[snapped].to(torch.long)]
        out[0][snapped] = csel[:, 0]
        out[1][snapped] = csel[:, 1]
        out[2][snapped] = csel[:, 2]

    # --- Step 2: fallback around voxel location ---
    unresolved = ~snapped
    if unresolved.any():
        base_z = torch.arange(D, device=device).view(D, 1, 1).expand(D, H, W)
        base_y = torch.arange(H, device=device).view(1, H, 1).expand(D, H, W)
        base_x = torch.arange(W, device=device).view(1, 1, W).expand(D, H, W)

        pending2 = unresolved.clone()
        for dz, dy, dx in nb_offsets:
            if not pending2.any():
                break
            zz = (base_z + dz).clamp(0, D - 1)
            yy = (base_y + dy).clamp(0, H - 1)
            xx = (base_x + dx).clamp(0, W - 1)
            cand = idx_grid[zz, yy, xx]
            hit = (cand >= 0) & pending2
            if hit.any():
                chosen[hit] = cand[hit]
                pending2[hit] = False

        picked = (chosen >= 0) & unresolved
        if picked.any():
            csel = centers_f[chosen[picked].to(torch.long)]
            out[0][picked] = csel[:, 0]
            out[1][picked] = csel[:, 1]
            out[2][picked] = csel[:, 2]

    return out



def assign_ids_by_hungarian(
    OriginalCenters: torch.Tensor,  # [N, 4] = [CellID, z, y, x]
    NewCenters: torch.Tensor,       # [M, 4] = [count,  z, y, x]
    metric: str = "chebyshev",      # "chebyshev" | "manhattan" | "euclidean"
    max_radius: float | None = None # if set, ignore matches farther than this
):
    """
    Returns:
      assigned:  [M, 4], same as NewCenters but column 0 replaced with matched CellID (or -1 if no match)
      row_ind:   1D tensor of matched NewCenters indices (size K = min(M,N))
      col_ind:   1D tensor of matched OriginalCenters indices (size K)
      dists:     1D tensor of distances for the K matches (same order as row_ind/col_ind)
    """
    device = NewCenters.device
    dtype_ids = OriginalCenters[:, 0].dtype

    # Extract coordinates
    new_xyz  = NewCenters[:, 1:].to(torch.float32)      # [M,3]
    orig_xyz = OriginalCenters[:, 1:].to(torch.float32) # [N,3]

    # Pairwise diffs: [M,N,3]
    diffs = (new_xyz[:, None, :] - orig_xyz[None, :, :]).abs()

    # Distance matrix [M,N]
    if metric == "chebyshev":
        cost = diffs.max(dim=2).values
    elif metric == "manhattan":
        cost = diffs.sum(dim=2)
    elif metric == "euclidean":
        cost = torch.sqrt((diffs ** 2).sum(dim=2))
    else:
        raise ValueError("metric must be 'chebyshev' | 'manhattan' | 'euclidean'")

    # Optionally block matches beyond max_radius
    if max_radius is not None:
        BIG = torch.finfo(cost.dtype).max / 4  # large sentinel cost
        cost = cost.clone()
        cost[cost > max_radius] = BIG

    # Hungarian on CPU numpy (SciPy)
    row_ind_np, col_ind_np = linear_sum_assignment(cost.detach().cpu().numpy())
    row_ind = torch.from_numpy(row_ind_np).to(device)
    col_ind = torch.from_numpy(col_ind_np).to(device)

    # Distances for chosen pairs
    dists = cost[row_ind, col_ind]

    # If max_radius used, drop assignments that exceeded it
    if max_radius is not None:
        ok = dists <= max_radius
        row_ind, col_ind, dists = row_ind[ok], col_ind[ok], dists[ok]

    # Build output: replace 'count' with matched CellID (unmatched → -1)
    assigned = NewCenters.clone()
    out_ids = torch.full((NewCenters.size(0),), -1, dtype=dtype_ids, device=device)
    out_ids[row_ind] = OriginalCenters[col_ind, 0]
    assigned[:, 0] = out_ids  # replace count with matched ID

    return assigned, row_ind, col_ind, dists

def ids_at_pointed_targets(NewCenters: torch.Tensor,
                                      NewCoords: torch.Tensor, valid: torch.Tensor,
                                      chunk_voxels: int = 200_000) -> torch.Tensor:
    """
    Returns a 1-channel tensor [Z, Y, X] of CellIDs.
    For each voxel's pointed-to coordinate in NewCoords (z,y,x),
    assigns the CellID of the nearest row in NewCenters ([CellID, z, y, x]).
    """
    device = NewCoords.device
    dtype  = NewCoords.dtype

    # shapes
    _, Z, Y, X = NewCoords.shape
    N = Z * Y * X

    # (N,3) in z,y,x order
    pts = NewCoords.reshape(3, -1).transpose(0, 1).contiguous().to(device)  # [N,3]

    # centers
    center_ids  = NewCenters[:, 0].to(torch.long).to(device)          # [K]
    centers_zyx = NewCenters[:, 1:4].to(dtype).to(device)             # [K,3]

    out_ids = torch.empty(N, dtype=torch.long, device=device)

    start = 0
    K = centers_zyx.shape[0]
    # Choose a chunk size that keeps (chunk_voxels x K) manageable
    # You can tune chunk_voxels based on K and VRAM.
    while start < N:
        end = min(start + chunk_voxels, N)
        p = pts[start:end]                                            # [B,3]

        # pairwise distances to centers: [B, K]
        # Broadcasting version (often faster than cdist for medium K):
        # d2 = ((p[:, None, :] - centers_zyx[None, :, :])**2).sum(-1)
        # Or use cdist (computes L2, argmin identical for squared):
        d = torch.cdist(p, centers_zyx, p=2)                          # [B,K]
        nn_idx = torch.argmin(d, dim=1)                               # [B]
        out_ids[start:end] = center_ids[nn_idx]

        start = end

    return (out_ids.view(Z, Y, X)).masked_fill(~valid, -1)  # [Z,Y,X]

def round_half_up(x: torch.Tensor) -> torch.Tensor:
    return torch.floor(x + 0.5)

def round_half_up_float(x):
    return floor(x + 0.5)

def voxel_points_to_self(targets: torch.Tensor, z: int, y: int, x: int, scale: float = 200.0) -> bool:
    # first center
    dz = targets[0, z, y, x].item()
    dy = targets[1, z, y, x].item()
    dx = targets[2, z, y, x].item()
    dist = targets[3, z, y, x].item() * scale

    cz = z + dz * dist
    cy = y + dy * dist
    cx = x + dx * dist

    zi, yi, xi = int(round(cz)), int(round(cy)), int(round(cx))

    Z, Y, X = targets.shape[1], targets.shape[2], targets.shape[3]
    if not (0 <= zi < Z and 0 <= yi < Y and 0 <= xi < X):
        return False

    # second center from the rounded voxel
    dz2 = targets[0, zi, yi, xi].item()
    dy2 = targets[1, zi, yi, xi].item()
    dx2 = targets[2, zi, yi, xi].item()
    dist2 = targets[3, zi, yi, xi].item() * scale

    cz2 = zi + dz2 * dist2
    cy2 = yi + dy2 * dist2
    cx2 = xi + dx2 * dist2

    zi2, yi2, xi2 = int(round(cz2)), int(round(cy2)), int(round(cx2))

    return (zi2 == zi) and (yi2 == yi) and (xi2 == xi)


def oneHotToDistance(source, centers):

    source[source < 0] = 0

    unique_vals, inverse = torch.unique(source, sorted=True, return_inverse=True)
    # inverse is a 1D tensor of indices in [0, K-1]; reshape it back:
    mapped = inverse.view(source.shape).long()            # same shape as source, now dense labels
    K = unique_vals.numel()

    # One-hot (channel-last), then move channel to front if you want (C, H, W) or (N, C, H, W)
    one_hot_last = F.one_hot(mapped, num_classes=K)

    one_hot_last = one_hot_last.squeeze(0).permute(3, 0, 1, 2)

    one_hot_last = one_hot_last[1:] 

    allMasks = (one_hot_last > 0)

        # 4-connected neighbors
    neighbors = [( -1 , -1 , -1 ),
        ( -1 , -1 , 0 ),
        ( -1 , -1 , 1 ),
        ( -1 , 0 , -1 ),
        ( -1 , 0 , 0 ),
        ( -1 , 0 , 1 ),
        ( -1 , 1 , -1 ),
        ( -1 , 1 , 0 ),
        ( -1 , 1 , 1 ),
        ( 0 , -1 , -1 ),
        ( 0 , -1 , 0 ),
        ( 0 , -1 , 1 ),
        ( 0 , 0 , -1 ),
        ( 0 , 0 , 0 ),
        ( 0 , 0 , 1 ),
        ( 0 , 1 , -1 ),
        ( 0 , 1 , 0 ),
        ( 0 , 1 , 1 ),
        ( 1 , -1 , -1 ),
        ( 1 , -1 , 0 ),
        ( 1 , -1 , 1 ),
        ( 1 , 0 , -1 ),
        ( 1 , 0 , 0 ),
        ( 1 , 0 , 1 ),
        ( 1 , 1 , -1 ),
        ( 1 , 1 , 0 ),
        ( 1 , 1 , 1 )]
    

    D,H,W = source.shape[1],source.shape[2],source.shape[3]
    finalDistances = torch.full((D, H, W), float("nan"))

    channelDistances = None

    for i in range(0, allMasks.shape[0]):
        D, H, W = allMasks[i].shape
        dist = torch.full((D, H, W), float("nan"))
        visited = torch.zeros((D, H, W), dtype=torch.bool)

        # Priority queue of (distance, z,y, x)
        heap = []
        ID, z0, y0, x0 = centers[i]

        if (ID != i+1):
            raise ValueError("ID != i+1")

        dist[int(z0), int(y0), int(x0)] = 0.0
        heapq.heappush(heap, (0.0, int(z0), int(y0), int(x0)))

        while heap:
            distance,z, y, x = heapq.heappop(heap)
            if visited[z,y, x]:
                continue
            visited[z,y, x] = True

            for dz, dy, dx in neighbors:
                nz, ny, nx = z+dz, y+dy, x+dx
                if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W and allMasks[i][nz, ny, nx] == 1:
                    new_d = distance + torch.sqrt(torch.tensor(dz*dz + dy*dy + dx*dx, dtype=torch.float32))
                    if new_d < dist[nz, ny, nx]:
                        dist[nz, ny, nx] = new_d
                        heapq.heappush(heap, (new_d.item(), nz, ny, nx))

        FDmask = torch.isnan(finalDistances)
        newDistMask = torch.isnan(dist)

        if (channelDistances == None):
            channelDistances = dist.clone()
        else:
            channelDistances = torch.cat((channelDistances, dist.clone()))

        finalDistances[FDmask & ~newDistMask] = 0
        dist[newDistMask & ~FDmask] = 0
        finalDistances = dist + finalDistances

    return finalDistances, channelDistances


import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

# -------- utilities --------

def _neighbors(connectivity=26, device=None, dtype=None):
    # 6 / 18 / 26 connectivity
    offs = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                steps = abs(dz) + abs(dy) + abs(dx)
                if connectivity == 6 and steps != 1:     # axis only
                    continue
                if connectivity == 18 and steps == 3:    # exclude body-diagonals
                    continue
                offs.append((dz, dy, dx))
    W = torch.tensor([(dx*dx + dy*dy + dz*dz) ** 0.5 for dz, dy, dx in offs],
                     device=device, dtype=dtype)
    return offs, W


def _shifted(t, dz, dy, dx, fill):
    # pad-and-slice shift without wraparound
    Z, Y, X = t.shape
    pad = (max(dx, 0), max(-dx, 0),
           max(dy, 0), max(-dy, 0),
           max(dz, 0), max(-dz, 0))
    tp = F.pad(t, pad, value=fill)
    z0 = max(-dz, 0); y0 = max(-dy, 0); x0 = max(-dx, 0)
    return tp[z0:z0+Z, y0:y0+Y, x0:x0+X]


# -------- core relaxer (uses +inf internally) --------

def _relax_geodesic(mask, seed_zyx, connectivity=26, max_iters=None):
    """
    mask: (Z,Y,X) bool tensor (True = allowed)
    seed_zyx: (z, y, x) int triplet within mask
    returns: (Z,Y,X) float distances (inf outside mask)  [internal]
    """
    device = mask.device
    dtype  = torch.float32
    Z, Y, X = mask.shape

    inf  = torch.tensor(float('inf'), device=device, dtype=dtype)
    dist = torch.full((Z, Y, X), float('inf'), device=device, dtype=dtype)

    z0, y0, x0 = map(int, seed_zyx)
    dist[z0, y0, x0] = 0.0

    offs, W = _neighbors(connectivity=connectivity, device=device, dtype=dtype)

    if max_iters is None:
        max_iters = 3 * max(Z, Y, X)  # heuristic upper bound

    for _ in range(max_iters):
        prev = dist
        cand = prev
        # Jacobi relax from neighbors; fill with +inf so minimum() propagates correctly
        for (dz, dy, dx), w in zip(offs, W):
            nbh = _shifted(prev, dz, dy, dx, fill=float('inf')) + w
            cand = torch.minimum(cand, nbh)

        # keep only inside the mask; outside remains +inf
        dist = torch.where(mask, cand, inf)

        # pin the seed
        dist[z0, y0, x0] = 0.0

        # convergence
        if torch.allclose(dist, prev, atol=1e-4, rtol=0.0):
            break

    return dist  # +inf outside mask


# -------- main function (uses +inf inside; converts to NaN on return) --------

def oneHotToDistance_fast(source, centers, connectivity=26):
    """
    source: labeled volume (D,H,W) or (1,D,H,W). Labels: 0=background, 1..K objects.
    centers: iterable of (ID, z, y, x) where ID corresponds to label value in `source`.
    connectivity: 6, 18, or 26

    returns:
      finalDistances (D,H,W)  -> NaN on background/outside
      channelDistances (K,D,H,W) -> NaN outside each object/channel
    """
    # --- normalize shape ---
    if source.ndim == 4 and source.shape[0] == 1:
        lab = source[0]
    elif source.ndim == 3:
        lab = source
    else:
        raise ValueError("source must be (D,H,W) or (1,D,H,W)")

    device = lab.device
    D, H, W = lab.shape

    # background mask (label <= 0)
    background = (lab <= 0)

    # Build masks per provided center ID (keep order)
    masks = []
    ids = []
    for (ID, z, y, x) in centers:
        ID = int(ID)
        ids.append(ID)
        m = (lab == ID)
        masks.append(m)

        # Optional safety: ensure seed lies inside its mask
        zi, yi, xi = int(z), int(y), int(x)
        if not (0 <= zi < D and 0 <= yi < H and 0 <= xi < W and m[zi, yi, xi]):
            raise ValueError(f"Seed ({zi},{yi},{xi}) not inside mask for ID={ID}")

    K = len(masks)

    # Internal representation: +inf outside, finite inside
    channelDistances = torch.full((K, D, H, W), float('inf'),
                                  device=device, dtype=torch.float32)
    finalDistances   = torch.zeros((D, H, W), device=device, dtype=torch.float32)

    for i, (ID, m) in enumerate(zip(ids, masks)):
        idx = m.nonzero(as_tuple=False)
        if idx.numel() == 0:
            continue

        zmin, ymin, xmin = idx.min(dim=0).values.tolist()
        zmax, ymax, xmax = (idx.max(dim=0).values + 1).tolist()
        m_roi = m[zmin:zmax, ymin:ymax, xmin:xmax]

        # seed (provided in absolute coords) -> ROI-local coords
        _, z0_abs, y0_abs, x0_abs = centers[i]
        z0 = int(z0_abs) - zmin
        y0 = int(y0_abs) - ymin
        x0 = int(x0_abs) - xmin

        # geodesic distance inside ROI (returns +inf outside ROI mask)
        dist_roi = _relax_geodesic(m_roi, (z0, y0, x0), connectivity=connectivity)

        # write back
        channelDistances[i, zmin:zmax, ymin:ymax, xmin:xmax] = dist_roi

        # accumulate only finite distances (ignore +inf)
        finite_roi = torch.isfinite(dist_roi)
        if finite_roi.any():
            finalDistances[zmin:zmax, ymin:ymax, xmin:xmax] += torch.where(
                finite_roi, dist_roi, torch.tensor(0.0, device=device, dtype=dist_roi.dtype)
            )

    # Background set to +inf (internal)
    finalDistances = torch.where(background, torch.tensor(float('inf'), device=device), finalDistances)

    # ---- convert sentinels (+inf) to NaN for outputs ----
    nan_scalar = torch.tensor(float('nan'), device=device)

    finalDistances = torch.where(torch.isfinite(finalDistances), finalDistances, nan_scalar)

    channelDistances = torch.where(torch.isfinite(channelDistances), channelDistances, nan_scalar)

    return finalDistances, channelDistances



@torch.no_grad()
def masked_gradient3d(
    vol: torch.Tensor,
    spacing=(1.0, 1.0, 1.0),
    out_nan_where_invalid: bool = True,
    inward_bias: float = 0.0,  # <-- NEW: set >0 for a nudge away from NaN sides
):
    """
    Accepts (C,D,H,W) or (D,H,W). Returns (C,3,D,H,W) grads [dz,dy,dx] and (C,D,H,W) valid mask.
    Ignores inf/NaN; central diff if both neighbors finite, else one-sided, else invalid.

    inward_bias:
      Adds a small directional nudge away from invalid neighbors.
      Example: if +x neighbor is NaN and -x neighbor is finite -> dx gets a negative bias.
      The bias per axis is: bias * ((m_{+axis} - m_{-axis}) / spacing_axis),
      applied only where the corresponding derivative is finite (so NaNs remain NaN).
    """
    if vol.ndim == 3:
        vol = vol.unsqueeze(0)  # -> (C,D,H,W)
    C, D, H, W = vol.shape
    sz, sy, sx = map(float, spacing)
    m = torch.isfinite(vol)

    def shift(t, dz, dy, dx, val):
        out = torch.empty_like(t)
        out.fill_(val)
        z0 = max(dz, 0); z1 = D + min(dz, 0); zs = slice(z0, z1)
        y0 = max(dy, 0); y1 = H + min(dy, 0); ys = slice(y0, y1)
        x0 = max(dx, 0); x1 = W + min(dx, 0); xs = slice(x0, x1)
        zs_src = slice(z0 - dz, z1 - dz)
        ys_src = slice(y0 - dy, y1 - dy)
        xs_src = slice(x0 - dx, x1 - dx)
        out[:, zs, ys, xs] = t[:, zs_src, ys_src, xs_src]
        return out

    # Neighbor values (NaN-filled) and masks (False-filled) at +/- 1 along each axis
    vpz, vmz = shift(vol, -1, 0, 0, float('nan')), shift(vol, 1, 0, 0, float('nan'))
    vpy, vmy = shift(vol, 0, -1, 0, float('nan')), shift(vol, 0, 1, 0, float('nan'))
    vpx, vmx = shift(vol, 0, 0, -1, float('nan')), shift(vol, 0, 0, 1, float('nan'))

    mpz, mmz = shift(m, -1, 0, 0, False), shift(m, 1, 0, 0, False)  # +z, -z
    mpy, mmy = shift(m, 0, -1, 0, False), shift(m, 0, 1, 0, False)  # +y, -y
    mpx, mmx = shift(m, 0, 0, -1, False), shift(m, 0, 0, 1, False)  # +x, -x

    def deriv(vp, vm, mp, mm, h):
        cen_ok = mp & mm & m
        fwd_ok = mp & m & ~mm
        bwd_ok = mm & m & ~mp
        out = torch.empty_like(vol, dtype=vol.dtype)
        out.fill_(float('nan') if out_nan_where_invalid else 0.0)
        out[cen_ok] = (vp[cen_ok] - vm[cen_ok]) / (2.0 * h)
        out[fwd_ok] = (vp[fwd_ok] - vol[fwd_ok]) / h
        out[bwd_ok] = (vol[bwd_ok] - vm[bwd_ok]) / h
        return out

    dz = deriv(vpz, vmz, mpz, mmz, sz)
    dy = deriv(vpy, vmy, mpy, mmy, sy)
    dx = deriv(vpx, vmx, mpx, mmx, sx)

    # Inward-bias nudge (only applied where derivative is finite)
    if inward_bias != 0.0:
        # Direction is (m_plus - m_minus): +1 if only +side valid, -1 if only -side valid, 0 otherwise.
        finite_dz = torch.isfinite(dz)
        finite_dy = torch.isfinite(dy)
        finite_dx = torch.isfinite(dx)

        bz = inward_bias * (mpz.to(vol.dtype) - mmz.to(vol.dtype)) / sz
        by = inward_bias * (mpy.to(vol.dtype) - mmy.to(vol.dtype)) / sy
        bx = inward_bias * (mpx.to(vol.dtype) - mmx.to(vol.dtype)) / sx

        dz = torch.where(finite_dz, dz + bz, dz)
        dy = torch.where(finite_dy, dy + by, dy)
        dx = torch.where(finite_dx, dx + bx, dx)

    grad = torch.stack([dz, dy, dx], dim=1)  # (C,3,D,H,W)
    valid = torch.isfinite(dz) & torch.isfinite(dy) & torch.isfinite(dx)  # (C,D,H,W)
    return grad, valid


def smoothDistance(distance, size=7, sigma=1.0):

    tmpDistance = distance.clone()
    MediumMask = torch.isnan(tmpDistance)
    MediumMask2 = MediumMask.clone()


    tmpDistance[MediumMask] = 0

    blurredDistance = blur3d(tmpDistance, sigma,size)

    MediumMask = (~MediumMask).to(torch.float32)

    blurredMask = blur3d(MediumMask, sigma,size)

    blurredMask[blurredMask == 0] = 1

    blurredNormalizedDistance = blurredDistance/blurredMask

    blurredNormalizedDistance[MediumMask2] = torch.nan

    return blurredNormalizedDistance

def blur3d(x: torch.Tensor, sigma: float = 1.0, size: int | None = None) -> torch.Tensor:
    """
    x: (C, D, H, W) float tensor
    sigma: Gaussian std
    size: kernel size (odd). If None, picks ~6*sigma+1.
    """
    if x.ndim == 3:
        C=1
    else:
        C, _, _, _ = x.shape

    if size is None:
        size = int(6 * sigma + 1)
    if size % 2 == 0:
        size += 1

    coords = torch.arange(size, device=x.device, dtype=torch.int16) - size // 2
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing="ij")  # correct axes
    g = torch.exp(-(xx*xx + yy*yy + zz*zz) / (2 * sigma * sigma))
    g /= g.sum()

    weight = g.view(1, 1, size, size, size).repeat(C, 1, 1, 1, 1)  # depthwise
    y = F.conv3d(x.unsqueeze(0), weight, padding=size // 2, groups=C)
    return y.squeeze(0)

def buildPiff(NewDistanceTensor: torch.Tensor,
              OriginalCenters: torch.Tensor,
              wallID,
              path: str,
              chunk_voxels: int = 2_000_000):
    """
    WRITE ONE ROW PER VOXEL with r >= 0.
    For each voxel (x,y,z): compute pointed center (cx,cy,cz) = (x+ux*r, y+uy*r, z+uz*r)
    and match that CONTINUOUS point to the nearest OriginalCenter (ID,Z,Y,X).
    Assign that ID to the voxel. No quantization, no NMS.
    Output columns (space-separated, no header):
        CellID CellType x1 x2 y1 y2 z1 z2
    """
    T = NewDistanceTensor
    if T.ndim == 5 and T.shape[1] == 4:       # [B,4,Z,Y,X]
        uz = T[0, 0].float()
        uy = T[0, 1].float()
        ux = T[0, 2].float()
        r  = T[0, 3].float()
        NewDistanceTensor= NewDistanceTensor.squeeze(0)
    elif T.ndim == 4 and T.shape[0] == 4:     # [4,Z,Y,X]
        uz = T[0].float()
        uy = T[1].float()
        ux = T[2].float()
        r  = T[3].float()
    else:
        raise ValueError("Expected NewDistanceTensor as (B,4,Z,Y,X) or (4,Z,Y,X) with [ux,uy,uz,r]")

    Z, Y, X = r.shape

    # keep EVERY voxel with r >= 0
    valid = (r >= 0)

    r = r*MAX_VOXEL_DIM

    r = r.masked_fill(~valid, -1)

    if not valid.any():
        open(path, "w").close()
        return 0


    CenterCoords = torch.zeros(
    (3, *NewDistanceTensor.shape[1:]),
    dtype=torch.float16,
    device=NewDistanceTensor.device
    )

    z_idx = torch.arange(Z, device=NewDistanceTensor.device).view(Z, 1, 1).expand(Z, Y, X)
    y_idx = torch.arange(Y, device=NewDistanceTensor.device).view(1, Y, 1).expand(Z, Y, X)
    x_idx = torch.arange(X, device=NewDistanceTensor.device).view(1, 1, X).expand(Z, Y, X)

    CenterCoords[0] = round_half_up(NewDistanceTensor[0] * NewDistanceTensor[3] * MAX_VOXEL_DIM + z_idx).masked_fill(~valid, -1)
    CenterCoords[1] = round_half_up(NewDistanceTensor[1] * NewDistanceTensor[3] * MAX_VOXEL_DIM + y_idx).masked_fill(~valid, -1)
    CenterCoords[2] = round_half_up(NewDistanceTensor[2] * NewDistanceTensor[3] * MAX_VOXEL_DIM + x_idx).masked_fill(~valid, -1)


    # Rearrange to [N, 3], where N = Z*Y*X
    triplets = CenterCoords.permute(1, 2, 3, 0)[valid]


    # Get unique rows
    unique_triplets, counts = torch.unique(triplets, dim=0, return_counts=True)

    # Concatenate so each row is [count, z, y, x]
    result = torch.cat([counts.unsqueeze(1), unique_triplets], dim=1)

    centers = drop_nearby_by_count(result)

    centers, _,_,_ = assign_ids_by_hungarian(OriginalCenters,centers)

    FinalTensor = ids_at_pointed_targets(centers, CenterCoords, valid)


    # start fresh file
    """
    FinalTensor: [1,Z,Y,X], each voxel holds CellID or -1
    wallID: ID that marks 'Wall' voxels
    path: output file path
    """
    # Drop channel dim -> [Z,Y,X]
    arr = FinalTensor.squeeze(0)

    zs, ys, xs = valid.nonzero(as_tuple=True)

    # Gather CellIDs
    cids = arr[zs, ys, xs].tolist()

    # Assign type
    celltype = ["Wall" if int(c) == int(wallID) else "Body" for c in cids]

    # Build DataFrame
    df = pd.DataFrame({
        "CellID": cids,
        "CellType": celltype,
        "x1": xs.tolist(), "x2": xs.tolist(),
        "y1": ys.tolist(), "y2": ys.tolist(),
        "z1": zs.tolist(), "z2": zs.tolist(),
    }, columns=["CellID","CellType","x1","x2","y1","y2","z1","z2"])

    df = df.sort_values(
    by=["x1", "x2", "y1", "y2", "z1", "z2"],
    ascending=True,  # or [True, False, ...] if you want different orders per column
    ignore_index=True  # optional: reset the index after sorting
    )

    # Write once (no chunk loop needed)
    df.to_csv(path, sep=" ", index=False, header=False)
    return df



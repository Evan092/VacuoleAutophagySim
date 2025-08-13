
import datetime
import torch
import argparse
import datetime
from functools import partial
import os
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
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
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt
import numpy as np
import datetime
import argparse
from Constants import *
from Discrimnator3D import Discriminator3D
from UNet3D import UNet3D
from CustomBatchSampler import CustomBatchSampler
from AccuracyTest import runAccuracyTest
from scipy.ndimage import zoom
from VoxelDataset import VoxelDataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from scipy.ndimage import maximum_filter
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import label

import torch
import torch.nn.functional as F

loadExisting = False
delete = False
saveLoaded = True

def pad_and_crop2(vol: np.ndarray, target_size: int = 200) -> np.ndarray:
    """
    vol: np.ndarray, shape (C, D, H, W), any C ≥ 1
    Returns a volume of shape (C, target_size, target_size, target_size)
    that is:
      1) Cropped to the tightest box containing any nonzero across all channels.
      2) Rescaled *isotropically* so its largest spatial dim == target_size.
      3) Padded equally on each side in the remaining axes to center it.
    """
    C, D, H, W = vol.shape

    # 1) find bounding‐box of any nonzero voxel across all channels
    occ = np.any(vol != 0, axis=0)  # shape (D,H,W)
    zs = np.where(np.any(occ, axis=(1,2)))[0]
    ys = np.where(np.any(occ, axis=(0,2)))[0]
    xs = np.where(np.any(occ, axis=(0,1)))[0]

    if zs.size and ys.size and xs.size:
        z0, z1 = zs[0], zs[-1] + 1
        y0, y1 = ys[0], ys[-1] + 1
        x0, x1 = xs[0], xs[-1] + 1
        vol = vol[:, z0:z1, y0:y1, x0:x1]

    # 2) isotropic rescale so max dimension == target_size
    _, D2, H2, W2 = vol.shape
    scale = target_size / max(D2, H2, W2)
    vol = zoom(vol, (1, scale, scale, scale), order=0)

    # 3) pad each axis to exactly target_size
    _, D3, H3, W3 = vol.shape
    pz = target_size - D3
    py = target_size - H3
    px = target_size - W3
    pzb, pza = pz // 2, pz - pz // 2
    pyb, pya = py // 2, py - py // 2
    pxb, pxa = px // 2, px - px // 2

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

    return vol.astype(np.float32)

import numpy as np

def pad_and_crop(vol: np.ndarray, target_size: int = 200) -> np.ndarray:
    """
    vol: np.ndarray, shape (C, D, H, W)
    Returns vol of shape (C, target_size, target_size, target_size) by:
      1) zero-padding smaller axes equally on both sides,
      2) center-cropping larger axes down to target_size,
      with NO scaling at all.
    """
    C, D, H, W = vol.shape

    # 1) Pad each axis up to target_size if it's smaller
    pad_z = max(target_size - D, 0)
    pad_y = max(target_size - H, 0)
    pad_x = max(target_size - W, 0)
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

    # 2) Crop each axis down to target_size if it's larger
    _, D2, H2, W2 = vol.shape
    cz = (D2 - target_size) // 2 if D2 > target_size else 0
    cy = (H2 - target_size) // 2 if H2 > target_size else 0
    cx = (W2 - target_size) // 2 if W2 > target_size else 0

    vol = vol[
        :,
        cz    : cz + target_size,
        cy    : cy + target_size,
        cx    : cx + target_size,
    ]

    return vol.astype(np.float32)


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
            cell_ID = int(cell_ID)
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

    target_size=200
    _, D2, H2, W2 = vol.shape
    if not (D2 == H2 == W2 == target_size):
        factors = (1, target_size / D2, target_size / H2, target_size / W2)
        vol = zoom(vol, zoom=factors, order=0)

    tensor = torch.from_numpy(vol)

    return tensor




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

    #vol = pad_and_crop(vol)

    target_size=200
    _, D2, H2, W2 = vol.shape
    if not (D2 == H2 == W2 == target_size):
        factors = (1, target_size / D2, target_size / H2, target_size / W2)
        vol = zoom(vol, zoom=factors, order=0)

    tensor = torch.from_numpy(vol)

    if saveLoaded:
        torch.save(tensor, os.path.join(str(path).removesuffix(".piff") + ".pt"))

    # Assume tensor has shape [1, D, H, W]
    body_mask = (tensor == 1).float()  # 1s where Body
    wall_mask = (tensor == 2).float()  # 1s where Wall
    Neither = (tensor == 0).float()  # 1s where Wall

    # Stack into 2-channel tensor: [2, D, H, W]
    tensor = torch.cat([body_mask, wall_mask, Neither], dim=0)
    vol = np.flip(vol, axis=3)
    return tensor





def matchInference(gt_Start, end, device):
    """
    gen_model:   your trained generator
    volumes:     torch.Tensor [1, C, D, H, W]
    gt_instances: dict mapping (ch, inst_id) → boolean np.array of shape (D,H,W)
    steps:       integer or tensor
    """
    with torch.no_grad():
        # 1) Generate
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))



        # 2) Post-process to hard labels
        pred_label = end.cpu().numpy().argmax(axis=0)  # [D,H,W]
        Z, Y, X = pred_label.shape

        # 3) Extract instance IDs via watershed per channel
        pred_ids = np.zeros((2, Z, Y, X), dtype=np.int32)
        for ch in (0, 1):
            mask = (pred_label == ch)
            dist = distance_transform_edt(mask).astype(np.float32)

            if ch == 0:
                # Bodies → one seed per CC, then watershed
                cc     = label(mask)                       # connected components
                props  = regionprops(cc)                   # get region props
                markers = np.zeros_like(cc, dtype=np.int32)
                for prop in props:
                    # round centroid to nearest integer voxel
                    zc, yc, xc = map(int, prop.centroid)
                    markers[zc, yc, xc] = prop.label

                labels_ws = watershed(-dist, markers, mask=mask)

            else:
                # Walls → simple CC‐label since they don’t need splitting
                labels_ws, _ = label(mask, return_num=True)

            pred_ids[ch] = labels_ws.astype(np.int32)

        # 4) Fast matching per channel using sparse contingency
        all_matches = []
        for ch in (0, 1):
            # build gt_ids array of shape [D,Y,X]
            gt_id_map = np.zeros((Z, Y, X), dtype=np.int32)
            gt_keys = [k for k in gt_Start if k[0] == ch]
            if not gt_keys:
                continue
            max_gt   = int(max(k[1] for k in gt_keys)) 
            for (_, inst_id), mask in gt_Start.items():
                if _ == ch:
                    gt_id_map[mask] = inst_id

            pred_id_map = pred_ids[ch]
            max_pred = int(pred_id_map.max())

            # flatten
            g = gt_id_map.ravel()
            p = pred_id_map.ravel()
            # only non-zero
            nz = (g > 0) & (p > 0)
            g_nz = g[nz]
            p_nz = p[nz]
            data = np.ones_like(g_nz, dtype=np.int32)
            C = sparse.coo_matrix((data, (g_nz, p_nz)),
                                   shape=(max_gt+1, max_pred+1)).tocsc()
            C = C[1:, 1:]  # drop background row/col

            # intersection counts
            I = C.toarray()  # shape (max_gt, max_pred)
            # per-instance areas
            
            flat_gt  = gt_id_map.ravel()
            counts   = np.bincount(flat_gt, minlength=max_gt+1)
            area_gt  = counts[1:].astype(np.float32)


            # after you’ve built pred_id_map and have max_pred as an int:

            flat_pred = pred_id_map.ravel()                                 # shape (D*H*W,)
            counts    = np.bincount(flat_pred, minlength=max_pred+1)        # length = max_pred+1
            area_pred = counts[1:].astype(np.float32)                       # drop background slot

            # union = a + b - intersection
            U = area_gt[:,None] + area_pred[None,:] - I
            cost = - (I / (U + 1e-9))

            # Hungarian matching
            row_idx, col_idx = linear_sum_assignment(cost)
            for r, c in zip(row_idx, col_idx):
                iou = -cost[r, c]
                all_matches.append(((ch, r+1), (ch, c+1), iou))

        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
        return pred_label, all_matches, pred_ids
    
def drop_small(pred_ids, min_size=10):
    cleaned = np.zeros_like(pred_ids)
    for ch in (0,1):
        mask = pred_ids[ch]>0
        cc, n = label(mask)
        counts = np.bincount(cc.ravel())
        # keep only labels with enough voxels
        keep = set(np.nonzero(counts>=min_size)[0])
        for lbl in keep:
            cleaned[ch][cc==lbl] = lbl
    return cleaned

def matchInference2(gt_Start, end, device):
    running_loss = 0.0
    running_gen_loss = 0.0
    running_real_loss = 0.0
    running_fake_loss = 0.0
    running_adv_loss = 0.0
    total = 0


    end = end.to(device)

    with torch.no_grad():

        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
        #Post-Process
        pred_label = end.cpu().numpy().argmax(axis=0)  # [Z,Y,X]
        Z, Y, X   = pred_label.shape

        # 2) Build pred_ids via watershed on each channel
        pred_ids = np.zeros((3, Z, Y, X), dtype=np.int32)
        for ch in (0, 1):
            mask = (pred_label == ch)

            # 1) Distance transform
            dist = distance_transform_edt(mask).astype(np.float32)


            # 2) Find peak coordinates (no 'indices' argument)
            peak_coords = peak_local_max(
                dist,
                min_distance=15,
                footprint=np.ones((3,3,3)),
                num_peaks=len(gt_Start)-1 if ch == 0 else 1,
                labels=mask
            )
            # peak_coords is an array of shape (N_peaks, 3)

            # 3) Build a markers image: each peak becomes a unique integer label
            markers = np.zeros_like(dist, dtype=np.int32)
            for idx, (z,y,x) in enumerate(peak_coords, start=1):
                markers[z, y, x] = idx

            # 4) Run watershed
            if ch != 0:
                labels_ws = label(mask)
            else:
                labels_ws = watershed(-dist, markers, mask=mask)
            pred_ids[ch] = labels_ws
        # 3) Match against GT as before
        all_matches = []
        for ch in [0,1]:
            # build dict of predicted instance masks
            pred_instances = {
                (ch, inst_id): (pred_ids[ch] == inst_id)
                for inst_id in range(1, int(pred_ids[ch].max()) + 1)
            }

            gt_keys   = [k for k in gt_Start if k[0] == ch]
            pred_keys = list(pred_instances)

            cost = np.zeros((len(gt_keys), len(pred_keys)), dtype=np.float32)
            for i, gk in enumerate(gt_keys):
                gmask = gt_Start[gk]
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


import numpy as np
from scipy.ndimage import label as cc_label, distance_transform_edt
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import datetime

def matchInference3(gt_Start, end, device,
                    min_cc_size=10,
                    lam=0.1,
                    min_distance=1):
    """
    gt_Start: dict[(ch, gt_id)] -> boolean mask (Z,Y,X)
    end:     torch.Tensor [1, C, Z, Y, X] logits or softmax
    """
    end = end.to(device)
    with torch.no_grad():
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))

        # Post-process to hard labels
        pred_label = end.cpu().numpy().argmax(axis=0)  # [Z,Y,X]
        Z, Y, X   = pred_label.shape

        # 1) Build raw pred_ids via watershed/CC per channel
        raw_pred_ids = np.zeros((2, Z, Y, X), dtype=np.int32)
        for ch in (0, 1):
            mask = (pred_label == ch)

            dist = distance_transform_edt(mask).astype(np.float32)

            peak_coords = peak_local_max(
                dist,
                min_distance=min_distance,
                footprint=np.ones((3,3,3)),
                num_peaks_per_label=len(gt_Start)-1 if ch == 0 else 1,
                labels=mask
            )

            markers = np.zeros_like(dist, dtype=np.int32)
            for idx, (z,y,x) in enumerate(peak_coords, start=1):
                markers[z, y, x] = idx

            if ch == 0:
                labels_ws = watershed(-dist, markers, mask=mask)
            else:
                labels_ws, _ = cc_label(mask)

            raw_pred_ids[ch] = labels_ws

        # 2) Filter out tiny CCs
        pred_ids = np.zeros_like(raw_pred_ids)
        for ch in (0,1):
            lab = raw_pred_ids[ch]             # e.g. [0,2,2,0,5,5,5,…]
            labels, counts = np.unique(lab, return_counts=True)
            keep = []
            for lbl_val, cnt in zip(labels, counts):
                if lbl_val == 0:
                    continue
                # for bodies keep everything ≥1 voxel; for walls apply min_cc_size
                if (ch == 0 and cnt >= 1) or (ch == 1 and cnt >= min_cc_size):
                    keep.append(lbl_val)
            mask = np.isin(lab, keep)         # True for voxels in kept labels
            pred_ids[ch] = lab * mask  

        # 3) Prepare instance lists and centroids
        gt_keys = [k for k in gt_Start if k[0] in (0,1)]
        pred_keys = [
            (ch, inst)
            for ch in (0,1)
            for inst in np.unique(pred_ids[ch])
            if inst != 0
        ]

        # compute centroids
        gt_centroids = {
            gk: np.array(np.where(gt_Start[gk])).mean(axis=1)
            for gk in gt_keys
        }
        pred_centroids = {}
        pred_instances = {}
        for pk in pred_keys:
            mask = pred_ids[pk[0]] == pk[1]
            pred_instances[pk] = mask
            pred_centroids[pk] = np.array(np.where(mask)).mean(axis=1)

        # 4) Build combined cost matrix
        max_dist = np.linalg.norm([Z, Y, X])
        cost = np.zeros((len(gt_keys), len(pred_keys)), dtype=np.float32)
        for i, gk in enumerate(gt_keys):
            gmask = gt_Start[gk]
            gcent = gt_centroids[gk] / max_dist
            for j, pk in enumerate(pred_keys):
                pmask = pred_instances[pk]
                inter = np.logical_and(gmask, pmask).sum()
                union = np.logical_or(gmask, pmask).sum()
                iou = inter / (union + 1e-9)
                pcent = pred_centroids[pk] / max_dist
                dist  = np.linalg.norm(gcent - pcent)
                cost[i, j] = -iou + lam * dist

        # 5) Hungarian assignment
        row_idx, col_idx = linear_sum_assignment(cost)
        all_matches = []
        for r, c in zip(row_idx, col_idx):
            gt_key   = gt_keys[r]
            pred_key = pred_keys[c]
            # report pure IoU, not the mixed cost
            inter = np.logical_and(gt_Start[gt_key], pred_instances[pred_key]).sum()
            union = np.logical_or(gt_Start[gt_key], pred_instances[pred_key]).sum()
            pure_iou = inter / (union + 1e-9)
            all_matches.append((gt_key, pred_key, float(pure_iou)))

        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
        return pred_label, all_matches, pred_ids



    total += end.size(0)


    print("Current_gen:", running_gen_loss/total)
    print("Current_real:", running_real_loss/total)
    print("Current_fake:", running_fake_loss/total)
    print("Current_adv:", running_adv_loss/total)
    print("Current:", running_loss/total)

    return running_loss / total


def buildPiff(pred_label, all_matches, pred_ids, path):
    rows = []
    Z, Y, X = pred_label.shape
    for (ch, gt_id), (ch2, pred_inst), iou in all_matches:
        assert ch == ch2
        mask = (pred_label == ch) & (pred_ids[ch] == pred_inst)
        zs, ys, xs = np.where(mask)
        cell_type = "Body" if ch == 0 else "Wall"
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
    df.to_csv(f"{path}", sep=' ', index=False, header=False)


def getInferenceData(path):

    # parse to numpy volume, convert to tensor with channel dim
    inputTensor = parse_voxel_file(path)
    
    #if self.transform:
     #   inputTensor = self.transform(inputTensor)

    IDTensor = parse_voxel_file_for_ID_matching(path)

    gt_instances = {}
    # channels: 0=body 1=wall, 2=medium (nothing)
    for ch in [0, 1]:
        # grab all the IDs in this channel (background is encoded as 0)
        ids = np.unique(IDTensor[ch])
        ids = ids[ids != 0]   # drop the 0 background
        for id_ in ids:
            # make a boolean mask for that specific cell (or wall)
            mask = (IDTensor[ch] == id_)
            gt_instances[(ch, id_)] = mask

    return inputTensor.unsqueeze(0), gt_instances

def buildPiffNoMatch(pred_label, path):
    """
    Build a PIFF where all body voxels (pred_label == 0) get CellID=5
    and all wall voxels (pred_label == 1) get CellID=6.
    """
    rows = []
    Z, Y, X = pred_label.shape

    # Find every Body voxel
    zs_b, ys_b, xs_b = np.where(pred_label == 0)
    for z, y, x in zip(zs_b, ys_b, xs_b):
        rows.append({
            "CellID":   5,
            "CellType": "Body",
            "x1":       int(x), "x2": int(x),
            "y1":       int(y), "y2": int(y),
            "z1":       int(z), "z2": int(z),
        })



    # Find every Wall voxel
    zs_w, ys_w, xs_w = np.where(pred_label == 1)
    for z, y, x in zip(zs_w, ys_w, xs_w):
        rows.append({
            "CellID":   6,
            "CellType": "Wall",
            "x1":       int(x), "x2": int(x),
            "y1":       int(y), "y2": int(y),
            "z1":       int(z), "z2": int(z),
        })

    # Assemble and write
    df = pd.DataFrame(rows, columns=[
        "CellID","CellType","x1","x2","y1","y2","z1","z2"
    ])
    df = df.sort_values(by=["x1","y1","z1"], ascending=True)
    df.to_csv(path, sep=' ', index=False, header=False)



noise_dim = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


piffNum = 0
start, matcher = getInferenceData("D:\\runs\\runs\\testRun\\output" + f"{(piffNum):03d}" + ".piff")
piffNum = piffNum + 50
end = parse_voxel_file("D:\\runs\\runs\\testRun\\output" + f"{(piffNum):03d}" + ".piff")
pred_label, all_matches, pred_ids = matchInference3(matcher, end, device)
buildPiff(pred_label, all_matches, pred_ids, "D:\\runs\\runs\\testRun\\AI2output" + f"{(piffNum):03d}" + ".piff")

start, matcher = getInferenceData("D:\\runs\\runs\\testRun\\Ai2output" + f"{(piffNum):03d}" + ".piff")
piffNum = piffNum + 50
end = parse_voxel_file("D:\\runs\\runs\\testRun\\output" + f"{(piffNum):03d}" + ".piff")
pred_label, all_matches, pred_ids = matchInference3(matcher, end, device)
buildPiff(pred_label, all_matches, pred_ids, "D:\\runs\\runs\\testRun\\AI2output" + f"{(piffNum):03d}" + ".piff")

start, matcher = getInferenceData("D:\\runs\\runs\\testRun\\Ai2output" + f"{(piffNum):03d}" + ".piff")
piffNum = piffNum + 50
end = parse_voxel_file("D:\\runs\\runs\\testRun\\output" + f"{(piffNum):03d}" + ".piff")
pred_label, all_matches, pred_ids = matchInference3(matcher, end, device)
buildPiff(pred_label, all_matches, pred_ids, "D:\\runs\\runs\\testRun\\AI2output" + f"{(piffNum):03d}" + ".piff")

start, matcher = getInferenceData("D:\\runs\\runs\\testRun\\Ai2output" + f"{(piffNum):03d}" + ".piff")
piffNum = piffNum + 50
end = parse_voxel_file("D:\\runs\\runs\\testRun\\output" + f"{(piffNum):03d}" + ".piff")
pred_label, all_matches, pred_ids = matchInference3(matcher, end, device)
buildPiff(pred_label, all_matches, pred_ids, "D:\\runs\\runs\\testRun\\AI2output" + f"{(piffNum):03d}" + ".piff")

start, matcher = getInferenceData("D:\\runs\\runs\\testRun\\Ai2output" + f"{(piffNum):03d}" + ".piff")
piffNum = piffNum + 50
end = parse_voxel_file("D:\\runs\\runs\\testRun\\output" + f"{(piffNum):03d}" + ".piff")
pred_label, all_matches, pred_ids = matchInference3(matcher, end, device)
buildPiff(pred_label, all_matches, pred_ids, "D:\\runs\\runs\\testRun\\AI2output" + f"{(piffNum):03d}" + ".piff")

start, matcher = getInferenceData("D:\\runs\\runs\\testRun\\Ai2output" + f"{(piffNum):03d}" + ".piff")
piffNum = piffNum + 50
end = parse_voxel_file("D:\\runs\\runs\\testRun\\output" + f"{(piffNum):03d}" + ".piff")
pred_label, all_matches, pred_ids = matchInference3(matcher, end, device)
buildPiff(pred_label, all_matches, pred_ids, "D:\\runs\\runs\\testRun\\AI2output" + f"{(piffNum):03d}" + ".piff")




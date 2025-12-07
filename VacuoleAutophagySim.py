import argparse
import datetime
from functools import partial
import gc
import os
import random
from matplotlib import pyplot as plt
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
from Utils import *
from ScheduleDropout import ScheduledDropout
from SigmaScheduler import ScheduledSigma
from VoxelDataset import VoxelDataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torch.amp import GradScaler
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

def set_requires_grad(model, req_grad):
    for p in model.parameters():
        p.requires_grad = req_grad

def gaussian_kernel3d(channels, kernel_size=5, sigma=1.0, device='cuda'):
    # build a separable 1D Gaussian, then outer‐product into 3D
    coords = torch.arange(kernel_size, device=device).float() - (kernel_size-1)/2
    g1d   = torch.exp(-(coords**2)/(2*sigma**2))
    g1d  /= g1d.sum()
    g3d   = g1d[:,None,None] * g1d[None,:,None] * g1d[None,None,:]      # (K,K,K)
    g3d   = g3d.unsqueeze(0).unsqueeze(0).repeat(channels,1,1,1,1)      # (C,1,K,K,K)
    return g3d

def blur_targets(targets, kernel_size=5, sigma=1.0):



    """
    targets:  Tensor of shape (B, C, D, H, W) with 0/1 values
    returns:  blurred float Tensor same shape
    """

    if sigma == 0 or kernel_size <= 1:
        return targets

    B,C,D,H,W = targets.shape
    kernel = gaussian_kernel3d(C, kernel_size, sigma, device=targets.device)
    # pad so output is same size
    pad = kernel_size // 2
    return F.conv3d(targets, kernel, padding=pad, groups=C)

def plotTensor(tensor, batchIdx=0, i=97, savePath=""):
    probs = tensor[:, 3:, ...]                 # [B, K, D, H, W] (or [B, K, H, W])

    idx = probs.argmax(dim=1, keepdim=True)         # [B, 1, ...]
    one_hot = torch.zeros_like(probs).scatter_(1, idx, 1)

    final = one_hot[:, 0:2, ...].sum(dim=1, keepdim=True).clamp_max(1) 

    restored = final.repeat(1, 3, 1, 1, 1)

    quiver_slice_zyx((tensor[:, :3, ...]*restored)[batchIdx],  axis='y', index=i, stride=1, savePath=savePath)


import torch

def check_grad_near_overflow(model, *, scaler=None, margin=1_000, optimizer=None):
    """
    Flags gradients whose absolute value is within `1/margin` of the dtype's max.
    Example: margin=1_000 => warn when |g| > finfo.max / 1_000.
    Works with AMP: pass GradScaler as `scaler` so grads get unscaled first.
    Returns: list of (name, max_abs, dtype, threshold)
    """
    # Unscale for AMP so you inspect true grad magnitudes
    if scaler is not None:
        scaler.unscale_(optimizer)  # or unscale_ on your optimizer earlier

    offenders = []
    for name, p in model.named_parameters():
        g = p.grad
        if g is None:
            continue
        # Use the tensor's computation dtype
        dt = g.dtype
        fi = torch.finfo(dt)
        threshold = fi.max / float(margin)

        gabs_max = g.detach().abs().max()
        # quick finite check (no assertion)
        if not torch.isfinite(gabs_max):
            offenders.append((name, float('inf'), str(dt), float(threshold)))
            continue

        if gabs_max.item() > threshold:
            offenders.append((name, gabs_max.item(), str(dt), float(threshold)))

    return offenders


def assert_all_params_finite(model):
    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            raise AssertionError(f"Non-finite value detected in parameter: {name}")





# ----------------------------------------
# 4) Main Script
# ----------------------------------------

skipNextDiscBackProp = False
skipNextGenBackProp = False
scaler = GradScaler()




















import torch
import torch.nn.functional as F

# --------- small helpers (no loss changes) ---------

def _normalize_dirs_nograd(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # NO GRAD normalizer for directions (for cost-building & D step)
    with torch.no_grad():
        out = t.clone()
        d = out[:, :3, ...]
        n = torch.linalg.norm(d, dim=1, keepdim=True).clamp_min(eps)
        out[:, :3, ...] = d / n
    return out

def _normalize_dirs_grad(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # GRAD-PRESERVING normalizer for directions (for G step)
    d = t[:, :3, ...]
    n = torch.linalg.norm(d, dim=1, keepdim=True).clamp_min(eps)
    d_norm = d / n
    return torch.cat([d_norm, t[:, 3:, ...]], dim=1)

def _cos_dir_loss_per_sample(pred_dirs, tgt_dirs, valid, weight, eps=1e-8):
    # your cosine-style gen_dir_loss but returns [B] (per-sample), not a scalar yet
    cos_sim   = (pred_dirs * tgt_dirs).sum(dim=1)            # [B,D,H,W]
    per_voxel = 1.0 - cos_sim
    num = (per_voxel * valid * weight).sum(dim=(1,2,3))               # [B]
    den = valid.sum(dim=(1,2,3)).clamp_min(1)                # [B]
    return num / den                                         # [B]

def _greedy_match(cost: torch.Tensor) -> torch.Tensor:
    """
    cost: [B,S,5] DETACHED. Returns a bool mask [B,S,5] selecting min(S,5) pairs
    per batch item (one-to-one greedy).
    """
    assert cost.ndim == 3
    B, S, K = cost.shape
    m = min(S, K)
    work = cost.clone()
    mask = torch.zeros_like(work, dtype=torch.bool)
    for _ in range(m):
        _, flat = work.view(B, -1).min(dim=1)  # [B]
        s = flat // K
        i = flat %  K
        b = torch.arange(B, device=cost.device)
        mask[b, s, i] = True
        work[b, s, :] = float('inf')
        work[b, :, i] = float('inf')
    return mask


def center_weight_mask(shape=(200,200,200), center=None, sigma=100.0, device='cuda'):
    D,H,W = shape
    if center is None:
        center = (D/2, H/2, W/2)
    z = torch.arange(D, device=device) - center[0]
    y = torch.arange(H, device=device) - center[1]
    x = torch.arange(W, device=device) - center[2]
    zz,yy,xx = torch.meshgrid(z,y,x, indexing='ij')
    r2 = zz**2 + yy**2 + xx**2
    mask = torch.exp(-r2/(2*sigma**2))
    return mask / mask.max() 

from PIL import Image

def nameTBD(gen_outputs, oldCenters, savePath=""):
    distToCenter = modelFlowToCenter(gen_outputs, iters=1000, step=0.5, mask_thresh=0.0, repel=0.2)
    coords_idx = displacements_to_coords(distToCenter, round_to_int=True)
    triplets = coords_idx.permute(1, 2, 3, 0)
    triplets = triplets[~torch.isnan(triplets[:,:,:,0]) & ~torch.isnan(triplets[:,:,:,1]) & ~torch.isnan(triplets[:,:,:,2])]
    triplets = triplets.reshape(-1,3)

    unique_triplets, counts = torch.unique(triplets, dim=0, return_counts=True) # how many map to each voxel
    
    result = torch.cat([counts.unsqueeze(1), unique_triplets], dim=1)
    
    centers = drop_nearby_by_count(result, radius=2.0, minCount=0)
    idx = torch.argsort(centers[:, 0], descending=True)
    centers_sorted = centers[idx]
    oldCenters = oldCenters.squeeze(0)
    centers_sorted = centers_sorted[:(len(oldCenters)+1), :]

    final_coords = snap_coords_fast(coords_idx, centers_sorted,
    r_snap=1,          # radius => includes ±3 in each axis
    r_neighbor=3,      # radius => includes ±3
    treat_zero_as_bg=True,
    interpret="radius")

    oldCenters = oldCenters.to(centers_sorted.device)
    newCentersWithIDs = hungarian_match_coordinates(centers_sorted, oldCenters.squeeze(0))


    ids = ids_from_centers(final_coords, newCentersWithIDs)
    rgb_slice = render_cluster_slice(ids, axis='z', index=100, background='black')

    if savePath != "":
        Image.fromarray(rgb_slice.cpu().numpy()).save(savePath+".png")
    else:
        plt.imshow(rgb_slice.cpu().numpy())  # rgb_slice is (H,W,3) uint8
        plt.axis('off')
        plt.show()




def nameTBD2(gen_outputs, expectedBodies, savePath=""):
    distToCenter = gen_outputs[:, :3, ...]
    coords_idx = displacements_to_coords(distToCenter, round_to_int=True)
    triplets = coords_idx.permute(1, 2, 3, 0)
    triplets = triplets[~torch.isnan(triplets[:,:,:,0]) & ~torch.isnan(triplets[:,:,:,1]) & ~torch.isnan(triplets[:,:,:,2])]
    triplets = triplets.reshape(-1,3)

    unique_triplets, counts = torch.unique(triplets, dim=0, return_counts=True) # how many map to each voxel
    
    result = torch.cat([counts.unsqueeze(1), unique_triplets], dim=1)
    
    centers = drop_nearby_by_count(result, radius=2.0, minCount=0)
    idx = torch.argsort(centers[:, 0], descending=True)
    centers_sorted = centers[idx]
    centers_sorted = centers_sorted[:expectedBodies, :]

    final_coords = snap_coords_fast(coords_idx, centers_sorted,
    r_snap=1,          # radius => includes ±3 in each axis
    r_neighbor=3,      # radius => includes ±3
    treat_zero_as_bg=True,
    interpret="radius")


    mediumMask = gen_outputs[:,1:,...]

    final_coords[mediumMask] = torch.nan




    ids = cluster_ids_from_coords(final_coords)

    return ids
    #rgb_slice = render_cluster_slice(ids, axis='y', index=100, background='black')

    #if savePath != "":
    #    Image.fromarray(rgb_slice.cpu().numpy()).save(savePath+".png")
    #else:
    #    plt.imshow(rgb_slice.cpu().numpy())  # rgb_slice is (H,W,3) uint8
    #    plt.axis('off')
    #    plt.show()





def modelOutputToVol(gen_outputs, oldCenters):
    distToCenter = modelFlowToCenter(gen_outputs, iters=1000, step=1.0, mask_thresh=0.0, repel=0.2)
    coords_idx = displacements_to_coords(distToCenter, round_to_int=True)
    triplets = coords_idx.permute(1, 2, 3, 0)
    triplets = triplets[~torch.isnan(triplets[:,:,:,0]) & ~torch.isnan(triplets[:,:,:,1]) & ~torch.isnan(triplets[:,:,:,2])]
    triplets = triplets.reshape(-1,3)

    unique_triplets, counts = torch.unique(triplets, dim=0, return_counts=True) # how many map to each voxel
    
    result = torch.cat([counts.unsqueeze(1), unique_triplets], dim=1)
    
    centers = drop_nearby_by_count(result, radius=2.0, minCount=0)
    idx = torch.argsort(centers[:, 0], descending=True)
    centers_sorted = centers[idx]
    oldCenters = oldCenters.squeeze(0)
    centers_sorted = centers_sorted[:(len(oldCenters)+1), :]

    final_coords = snap_coords_fast(coords_idx, centers_sorted,
    r_snap=1,          # radius => includes ±3 in each axis
    r_neighbor=3,      # radius => includes ±3
    treat_zero_as_bg=True,
    interpret="radius")

    oldCenters = oldCenters.to(centers_sorted.device)
    newCentersWithIDs = hungarian_match_coordinates(centers_sorted, oldCenters.squeeze(0))


    ids = ids_from_centers(final_coords, newCentersWithIDs)

    del distToCenter, coords_idx, triplets, unique_triplets,counts,result,centers,centers_sorted,final_coords
    return ids, newCentersWithIDs






# --------- MEMORY-EFFICIENT TRAIN (loss math unchanged) ---------

def trainOld2(gen_model, disc_model, dataloader,
          gen_optimizer, disc_optimizer,
          gen_dist_criterion, gen_dir_criterion,   # kept for signature parity
          disc_criterion, device, epochNumber):

    # externals used in your codebase:
    # - blur_targets
    # - set_requires_grad
    # - ScheduledDropout
    # - sigma_sched
    # - printAndLog
    global sigma_sched

    gen_model.train()
    disc_model.train()

    # weights exactly as you computed
    eps = 1e-8
    real_weight = 1.0
    fake_weight = 1.0
    
    gen_prob_weight = 5
    gen_dir_weight = 2.5
    adv_weight = 3

    # bookkeeping
    running_loss = 0.0
    running_gen_prob_loss = 0.0
    running_gen_dir_loss  = 0.0
    running_real_loss     = 0.0
    running_fake_loss     = 0.0
    running_adv_loss      = 0.0
    running_p1     = 0.0
    running_p2     = 0.0
    running_p3     = 0.0
    total = 0

    S = 5  # number of generator draws per x (same as your 5 rows)


    needToPrint = True
    for m in disc_model.modules():
        if isinstance(m, ScheduledDropout):
            if needToPrint:
                printAndLog(f"Dropout: {m.value:.4f}")
                needToPrint=False

    sigma_sched.step()
    printAndLog(f"New sigma: {sigma_sched.value:.4f}")

    for index, ((volumes, oldCenters, inputVol), (t1,c1, vol1),(t2,c2, vol2),(t3,c3, vol3),(t4,c4, vol4),(t5,c5, vol5), steps, path1) in enumerate(dataloader):
        if False:

            volumes = volumes.to(device)
            t1 = t1.to(device)
            gout = gen_model(volumes, steps[0].item())
            nameTBD(volumes, oldCenters, "C:\\Users\\evans\\New folder\\Original")
            nameTBD(t1, oldCenters, "C:\\Users\\evans\\New folder\\gt")
            nameTBD(gout, oldCenters, "C:\\Users\\evans\\New folder\\Model")
            del gout
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gout = gen_model(volumes, steps[0].item())
            nameTBD(gout, len(c2.squeeze(0))+1, (folder + "\\Trial " + str(2)))
            del gout
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gout = gen_model(volumes, steps[0].item())
            nameTBD(gout, len(c3.squeeze(0))+1, (folder + "\\Trial " + str(3)))
            del gout
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gout = gen_model(volumes, steps[0].item())
            nameTBD(gout, len(c4.squeeze(0))+1, (folder + "\\Trial " + str(4)))
            del gout
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gout = gen_model(volumes, steps[0].item())
            nameTBD(gout, len(c5.squeeze(0))+1, (folder + "\\Trial " + str(5)))
            del gout
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()



            folder = "C:\\Users\\evans\\Documents\\Independent Example Tests\\Examples A\\Example" + str(index)
            os.makedirs(folder, exist_ok=False)
            nameTBD(t1, len(c1.squeeze(0))+1, (folder + "\\Trial " + str(1)))
            del t1
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            nameTBD(t2, len(c2.squeeze(0))+1, (folder + "\\Trial " + str(2)))
            del t2
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            nameTBD(t3, len(c3.squeeze(0))+1, (folder + "\\Trial " + str(3)))
            del t3
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            nameTBD(t4, len(c4.squeeze(0))+1, (folder + "\\Trial " + str(4)))
            del t4
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            nameTBD(t5, len(c5.squeeze(0))+1, (folder + "\\Trial " + str(5)))
            del t5
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            continue




        print(index+1,"/",len(dataloader))
        volumes = volumes.to(device, non_blocking=True)
        B = volumes.size(0)
        targets_list = [t1.to(device, non_blocking=True),
                        t2.to(device, non_blocking=True),
                        t3.to(device, non_blocking=True),
                        t4.to(device, non_blocking=True),
                        t5.to(device, non_blocking=True)]
        inputVol = inputVol.to(device, non_blocking=True)

        vol_list = [vol1.to(device, non_blocking=True),
                        vol2.to(device, non_blocking=True),
                        vol3.to(device, non_blocking=True),
                        vol4.to(device, non_blocking=True),
                        vol5.to(device, non_blocking=True)]

        # ===== Prep targets ONCE (no grad), same normalization as your code =====
        # We create TWO views for each target:
        # - tn: normalized directions (what your code wrote back in-place)
        # - tprob_raw: the UN-BLURRED probabilities used by CE BEFORE you blur
        # - tprob_blur: blurred probabilities used later where you blur in your code
        T_norm = []
        T_prob_raw = []
        T_prob_blur = []
        T_valid = []
        with torch.no_grad():
            for t in targets_list:
                assert torch.isfinite(t).all()
                tn = t.clone()
                g  = tn[:, :3, ...]
                gn = torch.linalg.norm(g, dim=1, keepdim=True).clamp_min(eps)
                tn[:, :3, ...] = g / gn  # same as your in-place normalize

                valid = ((tn[:,3,...] == 1) | (tn[:,4,...] == 1)).to(tn.dtype)  # [B,D,H,W]
                tprob_raw  = tn[:, 3:, ...].clone()
                tprob_blur = blur_targets(tprob_raw, kernel_size=3, sigma=sigma_sched.value)

                T_norm.append(tn)
                T_prob_raw.append(tprob_raw)
                T_prob_blur.append(tprob_blur)
                T_valid.append(valid)

        # ===== PHASE 1: Build cost matrix [B,S,5] with NO GRAD (same losses) =====
        with torch.no_grad():
            preds_no_grad = []
            for _ in range(S):
                gout = gen_model(volumes, steps[0].item())        # [B,C,D,H,W]
                gout = _normalize_dirs_nograd(gout)               # your dir normalization
                preds_no_grad.append(gout)

            rows = []  # each row: [B,5]
            for s in range(S):
                ps = preds_no_grad[s]
                p_dir = ps[:, :3, ...]
                p_prb = ps[:, 3:, ...]
                row_terms = []
                for i in range(5):
                    midWeights = (center_weight_mask(T_norm[i].shape[-3:], sigma=50, device=T_norm[i].device)/2)+2
                    midWeights = midWeights.unsqueeze(0)
                    # gen_dir_loss — EXACTLY your formula
                    l_dir_ps = _cos_dir_loss_per_sample(p_dir, T_norm[i][:, :3, ...], T_valid[i], midWeights)  # [B]
                    # gen_prob_loss — EXACTLY your call (CE against UNBLURRED targets)
                    weights = torch.tensor([0.055, 0.07, 0.875], device=device)
                    l_prob_ps = F.cross_entropy(p_prb, T_prob_raw[i], label_smoothing=0.075, weight=weights, reduction='none')*midWeights

                    #ids = nameTBD(gen_outputs, oldCenters)

                    # reduction='none' returns [B, C?, ...]? CE expects class targets normally;
                    # your code uses distribution (one-hot / probs). Keep it: reduction='none' then mean over spatial dims.
                    # F.cross_entropy with probs target returns per-element loss; average over dims to [B]:
                    if l_prob_ps.ndim > 1:
                        l_prob_ps = l_prob_ps.mean(dim=tuple(range(1, l_prob_ps.ndim)))  # [B]

                    # adv_loss (generator-side) for cost: EXACTLY your disc_criterion(..., ones)
                    adv_out = disc_model(ps)
                    l_adv_ps = disc_criterion(adv_out, torch.ones_like(adv_out))
                    if l_adv_ps.ndim > 1:
                        l_adv_ps = l_adv_ps.mean(dim=tuple(range(1, l_adv_ps.ndim)))  # [B]

                    row_terms.append(gen_dir_weight*l_dir_ps + gen_prob_weight*l_prob_ps + adv_weight*l_adv_ps)  # [B]
                rows.append(torch.stack(row_terms, dim=1))  # [B,5]

            cost_mat = torch.stack(rows, dim=1).detach()  # [B,S,5]

        # ===== Matching on detached costs =====
        match_mask = _greedy_match(cost_mat)  # [B,S,5]
        if match_mask.sum() == 0:
            continue  # skip degenerate batch

        # ===== PHASE 2: Discriminator update (one backward) =====
        set_requires_grad(gen_model, False)
        set_requires_grad(disc_model, True)
        disc_optimizer.zero_grad(set_to_none=True)

        disc_terms = []
        with torch.no_grad():
            # precompute S fakes for D (following your exact fake pipeline)
            fakes_for_D = []
            for _ in range(S):
                gout = gen_model(volumes, steps[0].item())
                gout = _normalize_dirs_nograd(gout)
                fake_output = gout.detach().clone()
                idx = fake_output[:, -3:, ...].argmax(dim=1, keepdim=True)
                fake_output[:, -3:, ...] = torch.zeros_like(fake_output[:, -3:, ...]).scatter_(1, idx, 1.0)
                fakes_for_D.append(fake_output)

        # accumulate real/fake using matched pairs
        for s in range(S):
            for i in range(5):
                if not match_mask[:, s, i].any():
                    continue
                # REAL: your exact call
                disc_outputs_real = disc_model(T_norm[i])  # you passed 'targets' (which you had normalized in-place)
                real_loss = disc_criterion(disc_outputs_real, torch.full_like(disc_outputs_real, 0.85))
                # FAKE: your exact call (binarized channels, 0.15)
                disc_outputs_fake = disc_model(fakes_for_D[s])
                fake_loss = disc_criterion(disc_outputs_fake, torch.full_like(disc_outputs_fake, 0.15))
                term = (fake_loss * fake_weight) + (real_loss * real_weight)
                print(f"[Discriminator {len(disc_terms)+1}/5]")
                print(f"Real: {real_loss.item():.6f} [{(real_loss*real_weight).item():.6f}]")
                print(f"Fake: {fake_loss.item():.6f} [{(fake_loss*fake_weight).item():.6f}]")
                print(f"Total: {(real_loss + fake_loss).item():.6f} "
                    f"[{((real_loss*real_weight) + (fake_loss*fake_weight)).item():.6f}]")

                # reduce to scalar
                if term.ndim > 0:
                    term = term.mean()
                disc_terms.append(term)

        disc_loss = torch.stack(disc_terms).mean()
        disc_loss.backward()
        disc_optimizer.step()

        # ===== PHASE 3: Generator update (one backward) =====
        pairs_mask = match_mask.any(dim=0)
        num_terms  = int(pairs_mask.sum().item())
        set_requires_grad(disc_model, False)            # adv grads go into gen only
        set_requires_grad(gen_model, True)
        gen_optimizer.zero_grad(set_to_none=True)

        printed = 0
        for s in range(S):
            if not pairs_mask[s].any():
                continue

            # We recompute the forward for EACH matched (s,i). This trades a bit of compute for much lower memory.
            for i in range(5):
                if not pairs_mask[s, i]:
                    continue

                # forward WITH grad
                gen_outputs  = gen_model(volumes, steps[0].item())
                gen_outputs  = _normalize_dirs_grad(gen_outputs)     # grad-preserving normalize
                directions   = gen_outputs[:, :3, ...]
                probabilities= gen_outputs[:, 3:, ...]

                # (optional) restrict to batch items that matched this (s,i)
                bmask = match_mask[:, s, i]                           # [B] bool

                # ---- gen_dir_loss (same math) ----
                midWeights = (center_weight_mask(T_norm[i].shape[-3:], sigma=50, device=T_norm[i].device)/2)+2
                midWeights = midWeights.unsqueeze(0)
                dir_vec       = _cos_dir_loss_per_sample(directions, T_norm[i][:, :3, ...], T_valid[i], midWeights)  # [B]

                with torch.no_grad():
                    modelVol, calculatedCenters = modelOutputToVol(gen_outputs=gen_outputs, oldCenters=oldCenters)
                    modelVol = modelVol.unsqueeze(0)
                    p1 = volume_similarity_percent(modelVol, vol_list[i].squeeze(0))
                    p2 = aspect_similarity_percent(modelVol, vol_list[i].squeeze(0))
                    p3 = border_touch_similarity_percent(modelVol, vol_list[i].squeeze(0))

                    del modelVol, calculatedCenters, p1, p2, p3
                    torch.cuda.empty_cache()

                gen_dir_loss  = (dir_vec[bmask].mean() if bmask.any() else dir_vec.mean())               # scalar

                # ---- gen_prob_loss (memory-lean CE) ----
                logp          = torch.log_softmax(probabilities, dim=1)          # [B,C,D,H,W]
                valid         = T_prob_raw[i].sum(dim=1).clamp_max(1).to(logp.dtype)#T_valid[i].to(logp.dtype)                         # [B,D,H,W]
                ce_map        = -(T_prob_raw[i] * logp).sum(dim=1)  
                ce_map   = ce_map * midWeights              # [B,D,H,W]
                num           = (ce_map * valid).sum(dim=(1,2,3))                 # [B]
                den           = valid.sum(dim=(1,2,3)).clamp_min(1)               # [B]
                prob_vec      = num / den                                         # [B]
                gen_prob_loss = (prob_vec[bmask].mean() if bmask.any() else prob_vec.mean())

                # ---- adv loss (your exact form) ----
                adv_out   = disc_model(gen_outputs)
                adv_vec   = ((adv_out - torch.ones_like(adv_out))**2).mean(dim=tuple(range(1, adv_out.ndim)))  # [B]
                adv_loss  = (adv_vec[bmask].mean() if bmask.any() else adv_vec.mean())

                # ---- weighted sum, scale for accumulation, then immediate backward ----
                gterm = gen_dir_loss*gen_dir_weight + gen_prob_loss*gen_prob_weight + adv_loss*adv_weight
                (gterm / num_terms).backward()    # << no retain_graph; this frees the graph right away

                # 5-line print
                printed += 1
                print(f"[Generator {printed}/5]")
                print(f"Gen_dir:  {gen_dir_loss.item():.6f} [{(gen_dir_loss*gen_dir_weight).item():.6f}]")
                print(f"Gen_prob: {gen_prob_loss.item():.6f} [{(gen_prob_loss*gen_prob_weight).item():.6f}]")
                print(f"Adv:      {adv_loss.item():.6f} [{(adv_loss*adv_weight).item():.6f}]")
                print(f"Total:    {(gen_dir_loss + gen_prob_loss + adv_loss).item():.6f} "
                    f"[{(gen_dir_loss*gen_dir_weight + gen_prob_loss*gen_prob_weight + adv_loss*adv_weight).item():.6f}]")
                
                print(f"Volume Accuracy: {p1.item():.2f}%")
                print(f"Aspect Accuracy: {p2.item():.2f}%")
                print(f"Border Touch Accuracy: {p3.item():.2f}%")
                

                gdir  = float(gen_dir_loss.detach())
                gprob = float(gen_prob_loss.detach())
                gadv  = float(adv_loss.detach())

                running_gen_prob_loss += gprob * B
                running_gen_dir_loss  += gdir  * B
                running_adv_loss      += gadv  * B

                running_p1  += p1 * B
                running_p2  += p2  * B
                running_p3  += p3  * B

                running_loss += gprob + gdir + gadv

                # (optional) free temps early
                del gen_outputs, directions, probabilities, logp, ce_map, adv_out, modelVol, calculatedCenters

        gen_optimizer.step()

        # ===== Logging (same quantities) =====


        # recompute quick D real/fake for logs (cheap, no grad)
        with torch.no_grad():
            disc_real_log = disc_model(T_norm[0])
            real_loss_log = disc_criterion(disc_real_log, torch.full_like(disc_real_log, 0.85)).mean()
            fake_quick = _normalize_dirs_nograd(gen_model(volumes, steps[0].item()))
            idx = fake_quick[:, -3:, ...].argmax(dim=1, keepdim=True)
            fake_quick[:, -3:, ...] = torch.zeros_like(fake_quick[:, -3:, ...]).scatter_(1, idx, 1.0)
            disc_fake_log = disc_model(fake_quick)
            fake_loss_log = disc_criterion(disc_fake_log, torch.full_like(disc_fake_log, 0.15)).mean()

        running_loss          += (float(real_loss_log) + float(fake_loss_log)) * B
        running_real_loss     += float(real_loss_log) * B
        running_fake_loss     += float(fake_loss_log) * B
        total += B

    # ===== end-epoch prints / sched =====
    printAndLog("\n")
    printAndLog("Epoch_gen_prob:" + str( running_gen_prob_loss/total) + "[" + str( running_gen_prob_loss*gen_prob_weight/total) + "]" + "\n")
    printAndLog("Epoch_gen_dir:"  + str( running_gen_dir_loss /total) + "[" + str( running_gen_dir_loss *gen_dir_weight /total) + "]" + "\n")
    printAndLog("Epoch_real:"     + str( running_real_loss    /total) + "[" + str( running_real_loss    *1.0         /total) + "]" + "\n")
    printAndLog("Epoch_fake:"     + str( running_fake_loss    /total) + "[" + str( running_fake_loss    *1.0         /total) + "]" + "\n")
    printAndLog("Epoch_adv:"      + str( running_adv_loss     /total) + "[" + str( running_adv_loss     *adv_weight   /total) + "]" + "\n")

    printAndLog("Epoch_real:"     + str( running_p1    /total) + "[" + str( running_p1        /total) + "]" + "\n")
    printAndLog("Epoch_fake:"     + str( running_p2    /total) + "[" + str( running_p2         /total) + "]" + "\n")
    printAndLog("Epoch_adv:"      + str( running_p3     /total) + "[" + str( running_p3   /total) + "]" + "\n")

    avg_D_Loss  = ((running_real_loss + running_fake_loss)/total)/2 
    d_loss_diff = (abs(running_real_loss - running_fake_loss))/total
    avg_D_Loss_weighted  = ((running_real_loss*1.0 + running_fake_loss*1.0)/total)/2 
    d_loss_weighted_diff = (abs(running_real_loss*1.0 - running_fake_loss*1.0))/total

    printAndLog("avg_D_Loss:" + str(avg_D_Loss) + "[" + str( avg_D_Loss_weighted) + "]" + "\n")
    printAndLog("d_loss_diff:" + str(d_loss_diff) + "[" + str( d_loss_weighted_diff) + "]" + "\n")
    printAndLog("------------------------------------------" + "\n")

    needToPrint = True
    for m in disc_model.modules():
        if isinstance(m, ScheduledDropout):
            m.step()
            if needToPrint:
                printAndLog(f"New dropout: {m.value:.4f}")
                needToPrint=False

    sigma_sched.step()
    printAndLog(f"New sigma: {sigma_sched.value:.4f}")

    return running_loss / max(total, 1)





def train(gen_model, disc_model, dataloader,
          gen_optimizer, disc_optimizer,
          gen_dist_criterion, gen_dir_criterion,   # kept for signature parity
          disc_criterion, device, epochNumber,
          useAMP=False):

    import torch
    from torch.cuda.amp import autocast, GradScaler

    global sigma_sched

    gen_model.train()
    disc_model.train()

    eps = 1e-8
    real_weight = 1.0
    fake_weight = 1.0
    
    gen_prob_weight = 5
    gen_dir_weight  = 2.5
    adv_weight      = 3

    running_loss = 0.0
    running_gen_prob_loss = 0.0
    running_gen_dir_loss  = 0.0
    running_real_loss     = 0.0
    running_fake_loss     = 0.0
    running_adv_loss      = 0.0
    running_p1            = 0.0
    running_p2            = 0.0
    running_p3            = 0.0
    total = 0

    S = 5  # number of generator draws per x

    # AMP scalers
    scaler_disc = torch.amp.GradScaler(device=device.type, enabled=useAMP)
    scaler_gen  = torch.amp.GradScaler(device=device.type,enabled=useAMP)

    needToPrint = True
    for m in disc_model.modules():
        if isinstance(m, ScheduledDropout):
            if needToPrint:
                printAndLog(f"Dropout: {m.value:.4f}")
                needToPrint=False

    sigma_sched.step()
    printAndLog(f"New sigma: {sigma_sched.value:.4f}")

    # dataloader now yields PATHS ONLY
    # for index, ((oldPath),(t1Path),(t2Path),(t3Path),(t4Path),(t5Path), flips, steps)
    for index, ((oldPath),
                (t1Path),
                (t2Path),
                (t3Path),
                (t4Path),
                (t5Path),
                flips,
                steps) in enumerate(dataloader):

        print(index+1, "/", len(dataloader))

        # ============================================================
        # LOAD MAIN SAMPLE ONCE (volumes, oldCenters, inputVol)
        # ============================================================
        volumes, oldCenters, inputVol = loadData(oldPath, device, flips)
        volumes  = volumes.to(device, non_blocking=True)
        inputVol = inputVol.to(device, non_blocking=True)
        B = volumes.size(0)

        # ============================================================
        # PHASE 1: Build cost_mat [B,S,5] with NO GRAD
        # ============================================================
        with torch.no_grad():
            preds_no_grad = []
            for _s in range(S):
                gout = gen_model(volumes, steps[0].item())          # [B,C,D,H,W]
                gout = _normalize_dirs_nograd(gout)                 # dir norm (no grad)
                preds_no_grad.append(gout)

            rows = []  # length S, each [B,5]
            for s in range(S):
                ps    = preds_no_grad[s]        # [B,C,D,H,W]
                p_dir = ps[:, :3, ...]
                p_prb = ps[:, 3:, ...]
                row_terms = []

                # iterate each future target path
                for tPath in [t1Path, t2Path, t3Path, t4Path, t5Path]:
                    # load target now
                    t_full, c_full, vol_full = loadData(tPath, device, flips)
                    t_full = t_full.to(device, non_blocking=True)

                    # normalize dirs for this target
                    tn = t_full.clone()
                    g  = tn[:, :3, ...]
                    gn = torch.linalg.norm(g, dim=1, keepdim=True).clamp_min(eps)
                    tn[:, :3, ...] = g / gn

                    valid_mask = ((tn[:,3,...] == 1) | (tn[:,4,...] == 1)).to(tn.dtype)  # [B,D,H,W]
                    tprob_raw  = tn[:, 3:, ...].clone()  # [B,3,D,H,W]

                    midWeights = (center_weight_mask(
                        tn.shape[-3:],
                        sigma=50,
                        device=tn.device
                    )/2)+2
                    midWeights = midWeights.unsqueeze(0)  # [1,1,D,H,W] -> broadcast

                    # dir term
                    l_dir_ps = _cos_dir_loss_per_sample(
                        p_dir,
                        tn[:, :3, ...],
                        valid_mask,
                        midWeights
                    )  # [B]

                    # prob term (cross-entropy style)
                    weights = torch.tensor([0.055, 0.07, 0.875], device=device)
                    l_prob_ps = F.cross_entropy(
                        p_prb,
                        tprob_raw,
                        label_smoothing=0.075,
                        weight=weights,
                        reduction='none'
                    ) * midWeights

                    if l_prob_ps.ndim > 1:
                        l_prob_ps = l_prob_ps.mean(dim=tuple(range(1, l_prob_ps.ndim)))  # [B]

                    # adv term
                    adv_out   = disc_model(ps)
                    l_adv_ps  = disc_criterion(adv_out, torch.ones_like(adv_out))
                    if l_adv_ps.ndim > 1:
                        l_adv_ps = l_adv_ps.mean(dim=tuple(range(1, l_adv_ps.ndim)))     # [B]

                    row_terms.append(
                        gen_dir_weight*l_dir_ps
                        + gen_prob_weight*l_prob_ps
                        + adv_weight*l_adv_ps
                    )  # [B]

                    # free target tensors immediately
                    del t_full, c_full, vol_full
                    del tn, g, gn
                    del valid_mask, tprob_raw
                    del midWeights
                    del l_dir_ps, l_prob_ps, adv_out, l_adv_ps

                row_terms = torch.stack(row_terms, dim=1)  # [B,5]
                rows.append(row_terms)
                del ps, p_dir, p_prb, row_terms

            cost_mat = torch.stack(rows, dim=1).detach()  # [B,S,5]

            del rows, preds_no_grad

        # match
        match_mask = _greedy_match(cost_mat)  # [B,S,5]
        if match_mask.sum() == 0:
            del cost_mat, match_mask
            del volumes, oldCenters, inputVol
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            continue

        # ============================================================
        # PHASE 2: Discriminator update
        # ============================================================
        set_requires_grad(gen_model, False)
        set_requires_grad(disc_model, True)
        disc_optimizer.zero_grad(set_to_none=True)

        disc_terms = []

        # precompute S fakes for D (generator frozen, so no grad here)
        with torch.no_grad():
            fakes_for_D = []
            for _s in range(S):
                gout = gen_model(volumes, steps[0].item())
                gout = _normalize_dirs_nograd(gout)
                fake_output = gout.detach().clone()
                idx = fake_output[:, -3:, ...].argmax(dim=1, keepdim=True)
                fake_output[:, -3:, ...] = torch.zeros_like(fake_output[:, -3:, ...]).scatter_(1, idx, 1.0)
                fakes_for_D.append(fake_output)

        # now do disc forward/backward with grad
        with torch.amp.autocast(device_type=device.type, enabled=useAMP):
            for s in range(S):
                for i_idx, tPath in enumerate([t1Path, t2Path, t3Path, t4Path, t5Path]):
                    if not match_mask[:, s, i_idx].any():
                        continue

                    # REAL branch
                    t_full, c_full, vol_full = loadData(tPath, device, flips)
                    t_full = t_full.to(device, non_blocking=True)

                    tn = t_full.clone()
                    g  = tn[:, :3, ...]
                    gn = torch.linalg.norm(g, dim=1, keepdim=True).clamp_min(eps)
                    tn[:, :3, ...] = g / gn

                    disc_outputs_real = disc_model(tn)
                    real_loss = disc_criterion(
                        disc_outputs_real,
                        torch.full_like(disc_outputs_real, 0.85)
                    )

                    # FAKE branch
                    disc_outputs_fake = disc_model(fakes_for_D[s])
                    fake_loss = disc_criterion(
                        disc_outputs_fake,
                        torch.full_like(disc_outputs_fake, 0.15)
                    )

                    term = (fake_loss * fake_weight) + (real_loss * real_weight)

                    print(f"[Discriminator {len(disc_terms)+1}/{5*B}]")
                    print(f"Real: {real_loss.item():.6f} [{(real_loss*real_weight).item():.6f}]")
                    print(f"Fake: {fake_loss.item():.6f} [{(fake_loss*fake_weight).item():.6f}]")
                    print(
                        f"Total: {(real_loss + fake_loss).item():.6f} "
                        f"[{((real_loss*real_weight) + (fake_loss*fake_weight)).item():.6f}]"
                    )

                    if term.ndim > 0:
                        term = term.mean()
                    disc_terms.append(term)

                    del t_full, c_full, vol_full, tn, g, gn
                    del disc_outputs_real, disc_outputs_fake, real_loss, fake_loss, term

            disc_loss = torch.stack(disc_terms).mean()

        # backward + step for disc
        if useAMP:
            scaler_disc.scale(disc_loss).backward()
            scaler_disc.step(disc_optimizer)
            scaler_disc.update()
        else:
            disc_loss.backward()
            disc_optimizer.step()

        del disc_terms, fakes_for_D, disc_loss

        # ============================================================
        # PHASE 3: Generator update
        # ============================================================
        pairs_mask = match_mask.any(dim=0)   # [S,5]
        num_terms  = int(pairs_mask.sum().item())

        set_requires_grad(disc_model, False)
        set_requires_grad(gen_model, True)
        gen_optimizer.zero_grad(set_to_none=True)

        printed = 0
        for s in range(S):
            for i_idx, tPath in enumerate([t1Path, t2Path, t3Path, t4Path, t5Path]):
                if not pairs_mask[s, i_idx]:
                    continue

                # forward WITH grad
                with torch.amp.autocast(device_type=device.type, enabled=useAMP):
                    gen_outputs  = gen_model(volumes, steps[0].item())
                    gen_outputs  = _normalize_dirs_grad(gen_outputs)
                    directions   = gen_outputs[:, :3, ...]
                    probabilities= gen_outputs[:, 3:, ...]

                    bmask = match_mask[:, s, i_idx]  # [B] bool

                    # load GT for this target
                    t_full, c_full, vol_full = loadData(tPath, device, flips)
                    t_full    = t_full.to(device, non_blocking=True)
                    vol_full  = vol_full.to(device, non_blocking=True)

                    tn = t_full.clone()
                    g  = tn[:, :3, ...]
                    gn = torch.linalg.norm(g, dim=1, keepdim=True).clamp_min(eps)
                    tn[:, :3, ...] = g / gn

                    valid_mask = ((tn[:,3,...] == 1) | (tn[:,4,...] == 1)).to(tn.dtype)
                    tprob_raw  = tn[:, 3:, ...].clone()

                    midWeights = (center_weight_mask(
                        tn.shape[-3:],
                        sigma=50,
                        device=tn.device
                    )/2)+2
                    midWeights = midWeights.unsqueeze(0)

                    # dir loss
                    dir_vec = _cos_dir_loss_per_sample(
                        directions,
                        tn[:, :3, ...],
                        valid_mask,
                        midWeights
                    )  # [B]

                    # metrics (no grad)
                    with torch.no_grad():
                        modelVol, calculatedCenters = modelOutputToVol(
                            gen_outputs=gen_outputs,
                            oldCenters=oldCenters
                        )
                        modelVol = modelVol.unsqueeze(0)

                        p1 = volume_similarity_percent(modelVol, vol_full)
                        p2 = aspect_similarity_percent(modelVol, vol_full)
                        p3 = border_touch_similarity_percent(modelVol, vol_full)

                        p1_val = float(p1.detach())
                        p2_val = float(p2.detach())
                        p3_val = float(p3.detach())

                        del modelVol, calculatedCenters, p1, p2, p3
                        torch.cuda.empty_cache()

                    gen_dir_loss = (dir_vec[bmask].mean() if bmask.any() else dir_vec.mean())

                    # prob loss
                    logp   = torch.log_softmax(probabilities, dim=1)
                    valid2 = tprob_raw.sum(dim=1).clamp_max(1).to(logp.dtype)
                    ce_map = -(tprob_raw * logp).sum(dim=1)
                    ce_map = ce_map * midWeights

                    num = (ce_map * valid2).sum(dim=(1,2,3))
                    den = valid2.sum(dim=(1,2,3)).clamp_min(1)
                    prob_vec      = num / den
                    gen_prob_loss = (prob_vec[bmask].mean() if bmask.any() else prob_vec.mean())

                    # adv loss
                    adv_out = disc_model(gen_outputs)
                    adv_vec = ((adv_out - torch.ones_like(adv_out))**2).mean(
                        dim=tuple(range(1, adv_out.ndim))
                    )
                    adv_loss = (adv_vec[bmask].mean() if bmask.any() else adv_vec.mean())

                    gterm = gen_dir_loss*gen_dir_weight \
                            + gen_prob_loss*gen_prob_weight \
                            + adv_loss*adv_weight

                # backward for generator (scaled or not)
                if useAMP:
                    scaler_gen.scale(gterm / num_terms).backward()
                else:
                    (gterm / num_terms).backward()

                printed += 1
                print(f"[Generator {printed}/5]")

                gdir  = float(gen_dir_loss.detach())
                gprob = float(gen_prob_loss.detach())
                gadv  = float(adv_loss.detach())

                print(f"Gen_dir:  {gdir:.6f} [{(gdir*gen_dir_weight):.6f}]")
                print(f"Gen_prob: {gprob:.6f} [{(gprob*gen_prob_weight):.6f}]")
                print(f"Adv:      {gadv:.6f} [{(gadv*adv_weight):.6f}]")
                print(
                    f"Total:    {(gdir + gprob + gadv):.6f} "
                    f"[{(gdir*gen_dir_weight + gprob*gen_prob_weight + gadv*adv_weight):.6f}]"
                )
                print(f"Volume Accuracy: {p1_val:.2f}%")
                print(f"Aspect Accuracy: {p2_val:.2f}%")
                print(f"Border Touch Accuracy: {p3_val:.2f}%")

                running_gen_prob_loss += gprob * B
                running_gen_dir_loss  += gdir  * B
                running_adv_loss      += gadv  * B

                running_p1  += p1_val * B
                running_p2  += p2_val * B
                running_p3  += p3_val * B

                running_loss += gprob + gdir + gadv

                # free per-(s,i) stuff
                del gen_outputs, directions, probabilities
                del t_full, c_full, vol_full
                del tn, g, gn, valid_mask, tprob_raw
                del midWeights, dir_vec
                del logp, valid2, ce_map, num, den, prob_vec
                del adv_out, adv_vec, gen_dir_loss, gen_prob_loss, adv_loss, gterm
                torch.cuda.empty_cache()

        # step optimizer for generator
        if useAMP:
            scaler_gen.step(gen_optimizer)
            scaler_gen.update()
        else:
            gen_optimizer.step()

        # ============================================================
        # LOGGING / METRICS AT BATCH END
        # ============================================================
        with torch.no_grad():
            t_full_log, c_full_log, vol_full_log = loadData(t1Path, device, flips)
            t_full_log = t_full_log.to(device, non_blocking=True)

            tn_log = t_full_log.clone()
            g_log  = tn_log[:, :3, ...]
            gn_log = torch.linalg.norm(g_log, dim=1, keepdim=True).clamp_min(eps)
            tn_log[:, :3, ...] = g_log / gn_log

            disc_real_log = disc_model(tn_log)
            real_loss_log = disc_criterion(
                disc_real_log,
                torch.full_like(disc_real_log, 0.85)
            ).mean()

            fake_quick = _normalize_dirs_nograd(gen_model(volumes, steps[0].item()))
            idx = fake_quick[:, -3:, ...].argmax(dim=1, keepdim=True)
            fake_quick[:, -3:, ...] = torch.zeros_like(fake_quick[:, -3:, ...]).scatter_(1, idx, 1.0)
            disc_fake_log = disc_model(fake_quick)
            fake_loss_log = disc_criterion(
                disc_fake_log,
                torch.full_like(disc_fake_log, 0.15)
            ).mean()

            del t_full_log, c_full_log, vol_full_log
            del tn_log, g_log, gn_log
            del disc_real_log, disc_fake_log, fake_quick
            torch.cuda.empty_cache()

        running_loss      += (float(real_loss_log) + float(fake_loss_log)) * B
        running_real_loss += float(real_loss_log) * B
        running_fake_loss += float(fake_loss_log) * B
        total += B

        del real_loss_log, fake_loss_log
        del cost_mat, match_mask, pairs_mask
        del volumes, oldCenters, inputVol
        torch.cuda.empty_cache()

    # ===========================
    # end-epoch logging / sched
    # ===========================
    printAndLog("\n")
    printAndLog("Epoch_gen_prob:" + str(running_gen_prob_loss/total) +
                "[" + str(running_gen_prob_loss*gen_prob_weight/total) + "]" + "\n")
    printAndLog("Epoch_gen_dir:"  + str(running_gen_dir_loss /total) +
                "[" + str(running_gen_dir_loss *gen_dir_weight /total) + "]" + "\n")
    printAndLog("Epoch_real:"     + str(running_real_loss    /total) +
                "[" + str(running_real_loss    *1.0         /total) + "]" + "\n")
    printAndLog("Epoch_fake:"     + str(running_fake_loss    /total) +
                "[" + str(running_fake_loss    *1.0         /total) + "]" + "\n")
    printAndLog("Epoch_adv:"      + str(running_adv_loss     /total) +
                "[" + str(running_adv_loss     *adv_weight   /total) + "]" + "\n")

    printAndLog("Epoch_real:"     + str(running_p1    /total) +
                "[" + str(running_p1        /total) + "]" + "\n")
    printAndLog("Epoch_fake:"     + str(running_p2    /total) +
                "[" + str(running_p2        /total) + "]" + "\n")
    printAndLog("Epoch_adv:"      + str(running_p3    /total) +
                "[" + str(running_p3        /total) + "]" + "\n")

    avg_D_Loss  = ((running_real_loss + running_fake_loss)/total)/2 
    d_loss_diff = (abs(running_real_loss - running_fake_loss))/total
    avg_D_Loss_weighted  = ((running_real_loss*1.0 + running_fake_loss*1.0)/total)/2 
    d_loss_weighted_diff = (abs(running_real_loss*1.0 - running_fake_loss*1.0))/total

    printAndLog("avg_D_Loss:" + str(avg_D_Loss) +
                "[" + str(avg_D_Loss_weighted) + "]" + "\n")
    printAndLog("d_loss_diff:" + str(d_loss_diff) +
                "[" + str(d_loss_weighted_diff) + "]" + "\n")
    printAndLog("------------------------------------------" + "\n")

    needToPrint = True
    for m in disc_model.modules():
        if isinstance(m, ScheduledDropout):
            m.step()
            if needToPrint:
                printAndLog(f"New dropout: {m.value:.4f}")
                needToPrint=False

    sigma_sched.step()
    printAndLog(f"New sigma: {sigma_sched.value:.4f}")

    return running_loss / max(total, 1)












def trainOld(gen_model, disc_model, dataloader, gen_optimizer, disc_optimizer, gen_dist_criterion, gen_dir_criterion, disc_criterion, device, epochNumber):
    global skipNextDiscBackProp, skipNextGenBackProp
    global sigma_sched
    gen_model.train()
    disc_model.train()
    running_loss = 0.0
    running_gen_prob_loss = 0.0
    running_gen_dir_loss = 0.0
    running_real_loss = 0.0
    running_fake_loss = 0.0
    running_adv_loss = 0.0
    eps = 1e-8

    real_weight = 1
    fake_weight = 1

    adv_weight = 0.2
    gen_prob_weight = 3
    gen_dir_weight = 5.25

    weight_sum = adv_weight + gen_prob_weight + gen_dir_weight

    adv_weight = (adv_weight/weight_sum)*5
    gen_prob_weight = (gen_prob_weight/weight_sum)*5
    gen_dir_weight = (gen_dir_weight/weight_sum)*5

    total = 0

    skipThisGenBackProp = False
    skipThisDiscBackProp = False


    accum_steps = 1  # simulate batch_size = 4
    gen_optimizer.zero_grad(set_to_none=True)
    disc_optimizer.zero_grad(set_to_none=True)
    for i, (volumes, targets1,targets2,targets3,targets4,targets5, steps, path1) in enumerate(dataloader, start=1):
        volumes = volumes.to(device)
        targets1 = targets1.to(device)
        targets2 = targets2.to(device)
        targets3 = targets3.to(device)
        targets4 = targets4.to(device)
        targets5 = targets5.to(device)


        assert torch.isfinite(volumes).all()
        assert torch.isfinite(targets1).all()
        assert torch.isfinite(targets2).all()
        assert torch.isfinite(targets3).all()
        assert torch.isfinite(targets4).all()
        assert torch.isfinite(targets5).all()

        B, _, D, H, W = volumes.shape

        allTargets = (targets1,targets2,targets3,targets4,targets5)

        genLossesRow = []
        discLossesRow = []

        genLosses = []
        discLosses = []

        for _ in range(len(allTargets)):
            gen_outputs = gen_model(volumes, steps[0].item())

    
            directions = gen_outputs[:, :3, ...]

            probabilities = gen_outputs[:, 3:, ...] 

            p = directions.clone()
            p_norm = torch.linalg.norm(p, dim=1, keepdim=True).clamp_min(eps)
            gen_outputs[:, :3, ...] = p / p_norm

            for i in range(len(allTargets)):
                torch.cuda.empty_cache()
                targets = allTargets[i]

        #with autocast():  

                target_directions = targets[:, :3, ...]

                valid = ((targets[:,3,...] == 1) | (targets[:,4,...] == 1))
                
                target_probabilities = targets[:, 3:, ...]


                # Normalize the direction channels
                # bound raw logits
                g = target_directions.detach()       # no grad through targets

                # Compute norms safely
                
                g_norm = torch.linalg.norm(g, dim=1, keepdim=True).clamp_min(eps)

                # Replace *first three* channels with normalized unit vectors
                
                targets[:, :3, ...]     = g / g_norm
                m = valid.to(p.dtype)
                # cosine similarity per voxel -> [B,D,H,W]
                cos_sim = (gen_outputs[:, :3, ...] * targets[:, :3, ...]).sum(dim=1)

                # mask exactly like your SmoothL1 path
                per_voxel = 1.0 - cos_sim                   # 0 aligned, 2 opposite
                num = (per_voxel * m).sum(dim=(1,2,3))
                den = m.sum(dim=(1,2,3)).clamp_min(1)
                gen_dir_loss = (num / den).mean()






                #gen_dir_loss = F.smooth_l1_loss(directions[valid], target_directions[valid]).mean(dim=1).mean()
                #gen_dist_loss = gen_dist_criterion(gen_outputs[:, -1:, ...], targets[:, -1:, ...])


                gen_prob_loss = F.cross_entropy(probabilities, target_probabilities, label_smoothing=0.075)
                

                #target_idx = target_probabilities.argmax(dim=1).long()
                #gen_prob_loss = F.cross_entropy(probabilities, target_idx)

                target_probabilities = blur_targets(targets[:, 3:, ...], kernel_size=3, sigma=sigma_sched.value)
                
                
                blurred_targets = targets #blur_targets(targets, kernel_size=3, sigma=0.2)
                
                
                
                disc_outputs = disc_model(blurred_targets)

                real_loss = disc_criterion(disc_outputs, torch.full_like(disc_outputs, 0.85)) #0.9?


                fake_output = gen_outputs.detach()

                idx = fake_output[:, -3:, ...].argmax(dim=1, keepdim=True)        # [B,1,D,H,W]
                fake_output[:, -3:, ...] = torch.zeros_like(fake_output[:, -3:, ...]).scatter_(1, idx, 1.0)   

                #fake_output[:, :3] = fake_output[:, :3].masked_fill(~valid, 0)
                #fake_output[:, -1] = fake_output[:, -1].masked_fill(~valid, -1)

                disc_outputs = disc_model(fake_output)

                fake_loss = disc_criterion(disc_outputs, torch.full_like(disc_outputs, 0.15)) #0.1?


                #fake_output = torch.zeros_like(gen_outputs).scatter_(
                    #dim=1,
                    #index=gen_outputs.argmax(dim=1, keepdim=True),
                    #value=1.0)

                fake_output = gen_outputs




                #disc_optimizer.zero_grad()
                #scaler.scale((fake_loss * fake_weight) + (real_loss*real_weight)).backward()
                #scaler.step(disc_optimizer)
                #scaler.update()
                if True:
                    discLossesRow.append((fake_loss * fake_weight) + (real_loss * real_weight))
                else:
                    (((fake_loss * fake_weight) + (real_loss * real_weight))/accum_steps).backward()
                    if i % accum_steps == 0 or i == len(dataloader):
                        disc_optimizer.step()
                        print("Stepping Disc")
                        disc_optimizer.zero_grad(set_to_none=True)

                #disc_optimizer.step()
                

                disc_outputs2 = disc_model(fake_output)

                adv_loss = disc_criterion(disc_outputs2, torch.ones_like(disc_outputs2))

                
                assert torch.isfinite(gen_dir_loss).all()
                assert torch.isfinite(gen_prob_loss).all()
                assert torch.isfinite(adv_loss).all()
                #gen_optimizer.zero_grad()
                #scaler.scale((gen_dir_loss*gen_dir_weight) + (gen_prob_loss*gen_prob_weight) + (adv_loss*adv_weight)).backward()
                #scaler.step(gen_optimizer)
                #scaler.update()
                if True:
                    genLossesRow.append((gen_dir_loss*gen_dir_weight) + (gen_prob_loss*gen_prob_weight) + (adv_loss*adv_weight))
                else:
                    (((gen_dir_loss*gen_dir_weight) + (gen_prob_loss*gen_prob_weight) + (adv_loss*adv_weight))/accum_steps).backward()
                    if i % accum_steps == 0 or i == len(dataloader):
                        print("Stepping Gen")
                        gen_optimizer.step()
                        gen_optimizer.zero_grad(set_to_none=True)
                #gen_optimizer.step()
                

            genLosses.append(genLossesRow)
            discLosses.append(discLossesRow)

            genLossesRow = []
            discLossesRow = []


    genLosses = torch.stack([torch.stack(row, dim=0) for row in genLosses], dim=0)  # [5,5]
    discLosses = torch.stack([torch.stack(row, dim=0) for row in discLosses], dim=0)

    gen_loss, disc_loss = match_losses(genLosses, discLosses)

    set_requires_grad(gen_model, False)
    assert torch.isfinite(fake_loss).all()
    assert torch.isfinite(real_loss).all()
    disc_optimizer.zero_grad(set_to_none=True)
    disc_loss.backward()
    disc_optimizer.step()
    set_requires_grad(gen_model, True)

    set_requires_grad(disc_model, False)
    gen_optimizer.zero_grad(set_to_none=True)
    gen_loss.backward()
    gen_optimizer.step()
    set_requires_grad(disc_model, True)




    assert_all_params_finite(gen_model)
    assert_all_params_finite(disc_model)

    print("Steps: ", steps[0].item())
    print("Gen_prob_Loss: ", gen_prob_loss.item(), "[", (gen_prob_loss*gen_prob_weight).item(),"]")
    print("Gen_dir_Loss: ", gen_dir_loss.item(), "[", (gen_dir_loss*gen_dir_weight).item(),"]")
    print("real_Loss: ", real_loss.item(), "[", (real_loss*real_weight).item(),"]")
    print("fake_Loss: ", fake_loss.item(), "[", (fake_loss * fake_weight).item(),"]")
    print("adv_Loss: ", adv_loss.item(), "[", (adv_loss*adv_weight).item(),"]")

    running_loss += (gen_prob_loss.item() + gen_dir_loss.item() + real_loss.item() + fake_loss.item() + adv_loss.item()) * volumes.size(0)
    running_gen_prob_loss += (gen_prob_loss.item()) * volumes.size(0)
    running_gen_dir_loss += (gen_dir_loss.item()) * volumes.size(0)
    running_real_loss += (real_loss.item()) * volumes.size(0)
    running_fake_loss += (fake_loss.item()) * volumes.size(0)
    running_adv_loss += (adv_loss.item()) * volumes.size(0)
    total += volumes.size(0)


    printAndLog("\n")
    printAndLog("Epoch_gen_prob:" + str( running_gen_prob_loss/total) + "[" + str( running_gen_prob_loss*gen_prob_weight/total) + "]" + "\n")
    printAndLog("Epoch_gen_dir:" + str( running_gen_dir_loss/total) + "[" + str( running_gen_dir_loss*gen_dir_weight/total) + "]" + "\n")
    printAndLog("Epoch_real:" + str(  running_real_loss/total) + "[" + str( running_real_loss*real_weight/total) + "]" + "\n")
    printAndLog("Epoch_fake:" + str(  running_fake_loss/total) + "[" + str( running_fake_loss*fake_weight/total) + "]" + "\n")
    printAndLog("Epoch_adv:" + str(  running_adv_loss/total) + "[" + str( running_adv_loss*adv_weight/total) + "]" + "\n")
    avg_D_Loss  = ((running_real_loss + running_fake_loss)/total)/2 
    d_loss_diff = (abs(running_real_loss - running_fake_loss))/total

    avg_D_Loss_weighted  = ((running_real_loss*real_weight + running_fake_loss*fake_weight)/total)/2 
    d_loss_weighted_diff = (abs(running_real_loss*real_weight - running_fake_loss*fake_weight))/total
    printAndLog("avg_D_Loss:" + str(avg_D_Loss) + "[" + str( avg_D_Loss_weighted) + "]" + "\n")
    printAndLog("d_loss_diff:" + str(d_loss_diff) + "[" + str( d_loss_weighted_diff) + "]" + "\n")
    printAndLog("------------------------------------------" + "\n")

    needToPrint = True
    for m in disc_model.modules():
        if isinstance(m, ScheduledDropout):
            m.step()
            if needToPrint:
                printAndLog(f"New dropout: {m.value:.4f}")
                needToPrint=False

    sigma_sched.step()
    printAndLog(f"New sigma: {sigma_sched.value:.4f}")
    torch.cuda.empty_cache()
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


            fake_output = torch.zeros_like(gen_outputs).scatter(
                dim=1,
                index=gen_outputs.argmax(dim=1, keepdim=True),
                value=1.0)


            fake_output = fake_output.detach()

            disc_outputs = disc_model(fake_output)

            fake_loss = disc_criterion(disc_outputs, torch.full_like(disc_outputs, 0.1))


            fake_output = torch.zeros_like(gen_outputs).scatter(
                dim=1,
                index=gen_outputs.argmax(dim=1, keepdim=True),
                value=1.0)

            disc_outputs2 = disc_model(fake_output)

            adv_loss = disc_criterion(disc_outputs2, torch.ones_like(disc_outputs2))

            print(time.now())
            #Post-Process
            pred_label = gen_outputs.detach().cpu().numpy().argmax(axis=0)  # [Z,Y,X]

            all_matches = []
            for ch in [0,1]:
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

def runinference(gen_model, volumes, gt_instances, steps, device):
    """
    gen_model:   your trained generator
    volumes:     torch.Tensor [1, C, D, H, W]
    gt_instances: dict mapping (ch, inst_id) → boolean np.array of shape (D,H,W)
    steps:       integer or tensor
    """
    gen_model.eval()
    volumes = volumes.to(device)

    with torch.no_grad():
        # 1) Generate
        B = volumes.shape[0]
        z = torch.randn(B, noise_dim, device=device) * 0.1
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
        gen_outputs = gen_model(volumes, z, steps)
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))



        probs = torch.sigmoid(gen_outputs.squeeze(0))      # shape [2,D,H,W]
        p_body, p_wall = probs[0], probs[1]

        # a) decide which voxels are “neither”
        is_neither = (1-(p_body + p_wall) > p_body) & (1-(p_body + p_wall) > p_wall)

        # b) for the rest, pick the stronger channel
        rest      = ~is_neither
        rest_idx  = torch.argmax(probs[:, rest], dim=0)   # 0=body, 1=wall

        # c) build a final label map with 0=body,1=wall,2=neither
        tmpLabel = torch.empty(p_body.shape, dtype=torch.long, device=p_body.device)

        tmpLabel[is_neither] = 2
        tmpLabel[rest]      = rest_idx



        # 2) Post-process to hard labels
        pred_label = tmpLabel.cpu().numpy() #gen_outputs.squeeze(0).cpu().numpy().argmax(axis=0)  # [D,H,W]
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
            gt_keys = [k for k in gt_instances if k[0] == ch]
            if not gt_keys:
                continue
            max_gt   = int(max(k[1] for k in gt_keys)) 
            for (_, inst_id), mask in gt_instances.items():
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

def runinference2(gen_model, volumes, gt_instances, steps, device):
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

        gen_outputs = gen_model(volumes, z, steps)


        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
        #Post-Process
        pred_label = gen_outputs.squeeze(0).detach().cpu().numpy().argmax(axis=0)  # [Z,Y,X]
        Z, Y, X   = pred_label.shape

        # 2) Build pred_ids via watershed on each channel
        pred_ids = np.zeros((2, Z, Y, X), dtype=np.int32)
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
            if ch == 2:
                labels_ws, _ = label(mask)
            else:
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


def testProcessing(volumes, gt_instances, device):
    running_loss = 0.0
    running_gen_loss = 0.0
    running_real_loss = 0.0
    running_fake_loss = 0.0
    running_adv_loss = 0.0
    total = 0


    volumes = volumes.to(device)

    B = volumes.shape[0]
    z = torch.randn(B, noise_dim, device=device) * 0.1


    #Generate our predicted values

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))


    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
    #Post-Process
    pred_label = volumes.squeeze(0).detach().cpu().numpy().argmax(axis=0)  # [Z,Y,X]
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
        if ch == 2:
            labels_ws, _ = label(mask)
        else:
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

def getInferenceData(path):

    # parse to numpy volume, convert to tensor with channel dim
    inputTensor = parse_voxel_file(path)
    
    #if self.transform:
     #   inputTensor = self.transform(inputTensor)

    IDTensor = parse_voxel_file_for_ID_matching(path)

    gt_instances = {}
    # channels: 0=medium (ignore), 1=body, 2=wall
    for ch in [0, 1]:
        # grab all the IDs in this channel (background is encoded as 0)
        ids = np.unique(IDTensor[ch])
        ids = ids[ids != 0]   # drop the 0 background
        for id_ in ids:
            # make a boolean mask for that specific cell (or wall)
            mask = (IDTensor[ch] == id_)
            gt_instances[(ch, id_)] = mask

    return inputTensor.unsqueeze(0), gt_instances


import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

def _worker(file_path, device):
    # Returns are ignored just like your "_,_ = ..."
    parse_voxel_file_for_distance(file_path, device=device)
    return None
    

def preprocess_all(paths, device, num_workers, parallel=False):
    if (parallel):
        for idx in range(len(paths)):
            for outputNumber in range(5):
                for steps in range(6):
                    try:
                        print(idx,outputNumber,steps, " of ", len(paths), 5, 6)
                        output = f"outputs_{(outputNumber+1):02d}"
                        stepNumber=(steps+1)*50
                        _,_,_ = parse_voxel_file_for_distance(paths[idx] + "\\" + output + f"\\output{stepNumber:03d}.piff", device=device, parallel=parallel)
                    except Exception as e:
                        print("Error for " + paths[idx] + "\\" + output + f"\\output{stepNumber:03d}.piff" ) 
    

    # Build all file paths once (avoids work in workers)
    files = []
    for p in paths:
        for outputNumber in range(1, 6):        # 1..5
            output = f"outputs_{outputNumber:02d}"
            base = os.path.join(p, output)
            for steps in range(1, 7):           # 1..6
                stepNumber = steps * 50
                files.append(os.path.join(base, f"output{stepNumber:03d}.piff"))

    ctx = multiprocessing.get_context("spawn")   # safe on Windows
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
        futures = [ex.submit(_worker, fp, device) for fp in files]
        for f in as_completed(futures):
            # Will raise immediately if a worker failed
            f.result()


def printAndLog(myString):
    print(myString)
    with open("log.txt", "a") as f:
        f.write(myString + "\n")


def add_weight_decay(module, wd, skip_norm=True):
    decay, no_decay = [], []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        is_norm = any(k in name.lower() for k in ["norm", "bn", "gn", "ln"])
        if skip_norm and (is_norm or name.endswith(".bias")):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]

sigma_sched = None





def main():
    global sigma_sched

    parser = argparse.ArgumentParser()
    #Training related parameters
    parser.add_argument('--train', type=str, default="D:\\runs\\runs", help='Path to dataset') #
    parser.add_argument('--batchSize', type=int, default=1, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of Epochs to run')
    parser.add_argument('--gen_lr', type=float, default=4e-4, help='Learning rate for the Generator')
    parser.add_argument('--disc_lr', type=float, default=1e-4, help='Learning rate for the Discriminator')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use')
    parser.add_argument('--shuffle', type=bool, default=False, help='Whether to Shuffle the data')
    parser.add_argument('--pin_memory', type=bool, default=True, help='Whether to use pin_memory')
    parser.add_argument('--persistent_workers', type=bool, default=True, help='Whether to keep workers across each epoch')
    parser.add_argument('--preprocess_training_data', type=bool, default=False, help='Whether to preprocess the training data before training')


    #Inference related Parameters
    parser.add_argument('--steps', type=int, default=100, help='Number of steps')
    parser.add_argument('--frequency', type=int, default=10, help='Frequency file is saved')
    parser.add_argument('--input', type=str, default="", help='Input piff file for inferences')


    #Works for train or inference
    parser.add_argument('--gen_checkpoint', type=str, default="", help="Path to the generator's .pth checkpoint file")
    parser.add_argument('--disc_checkpoint', type=str, default="", help="Path to the discriminator's .pth checkpoint file")

    #Required
    parser.add_argument('--output', type=str, default="", help='Output Folder')


    args = parser.parse_args()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()



    printGpuCheck = True
    #Variables
    runMode = 0
    data_folder = args.train
    batch_size  = args.batchSize    # samples per GPU batch
    epochs      = args.epochs       # training duration
    gen_lr      = args.gen_lr       # generator learning rate
    disc_lr     = args.disc_lr      # discriminator learning rate
    preprocess = args.preprocess_training_data

    printAndLog("batch_size: " + str(batch_size))
    printAndLog("epochs to do: " + str(epochs))
    printAndLog("gen_lr: " + str(gen_lr))
    printAndLog("disc_lr: " + str(disc_lr))

    # Setup PyTorch device and data loader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (preprocess):
        print("Preprocessing piff data")
        paths = [
            os.path.join(data_folder, d)
            for d in os.listdir(data_folder)
            if (os.path.isdir(os.path.join(data_folder, d)) 
                and os.path.isdir(os.path.join(data_folder, d, "outputs_01"))
                and os.path.isdir(os.path.join(data_folder, d, "outputs_02"))
                and os.path.isdir(os.path.join(data_folder, d, "outputs_03"))
                and os.path.isdir(os.path.join(data_folder, d, "outputs_04"))
                and os.path.isdir(os.path.join(data_folder, d, "outputs_05"))
                and os.path.isfile(os.path.join(data_folder, d, "outputs_05", "output300.piff"))
                and not str(d).lower().endswith(".bat"))
        ]
        preprocess_all(paths, device, 2, parallel=True)




    dataset = VoxelDataset(data_folder)

    customBatchSampler = CustomBatchSampler(dataset, batch_size, shuffle=args.shuffle)

    loader  = DataLoader(
        dataset,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        batch_sampler=customBatchSampler,
        persistent_workers=args.persistent_workers,
        
    )

    # Instantiate model
    gen_model = UNet3D().to(device)
    disc_model = Discriminator3D().to(device)
    

    gen_state = None
    disc_state = None
    gen_epoch = 0
    disc_epoch = 0

    if args.gen_checkpoint == "": #No checkpoint specified, TODO use the latest in the checkpoint folder
        if os.path.exists("gen_check/latest.pth"):
            gen_state  = torch.load("gen_check/latest.pth",  map_location=device, weights_only=True)
    else:
        gen_state  = torch.load(args.gen_checkpoint,  map_location=device, weights_only=True)
    
    if args.disc_checkpoint == "":  #No checkpoint specified, TODO use the latest in the checkpoint folder
        if os.path.exists("disc_check/latest.pth"):
            disc_state = torch.load("disc_check/latest.pth", map_location=device, weights_only=True)
    else:
        disc_state = torch.load(args.disc_checkpoint, map_location=device, weights_only=True)
    


    if gen_state: #We loaded a checkpoint. (We wouldn't if it was the first run)
        gen_epoch = gen_state['epoch'] #Load the checkpoint the generator was on
        #gen_model.load_state_dict(gen_state['model_state_dict'], strict=False) #Load the weights and biases

        sg_in  = gen_state["model_state_dict"]           # what you saved
        sg_cur = gen_model.state_dict()            # what the model expects now

        # keep only keys that exist AND match shape
        sg_filt = {k: v for k, v in sg_in.items()
                if (k in sg_cur) and (v.shape == sg_cur[k].shape)}
        gen_model.load_state_dict(sg_filt, strict=False) #Load the weights and biases

    if disc_state: #We loaded a checkpoint. (We wouldn't if it was the first run)
        disc_epoch = disc_state['epoch'] #Load the checkpoint the discriminator was on
        

        sd_in  = disc_state["model_state_dict"]           # what you saved
        sd_cur = disc_model.state_dict()            # what the model expects now

        # keep only keys that exist AND match shape
        sd_filt = {k: v for k, v in sd_in.items()
                if (k in sd_cur) and (v.shape == sd_cur[k].shape)}
        disc_model.load_state_dict(sd_filt, strict=False) #Load the weights and biases


    gen_param_groups  = add_weight_decay(gen_model,  wd=3e-4)
    disc_param_groups = add_weight_decay(disc_model, wd=3e-4)
    #Create the optimizers
    gen_optimizer = torch.optim.AdamW(gen_param_groups, lr=gen_lr,betas=(0.5, 0.99))
    disc_optimizer = torch.optim.AdamW(disc_param_groups, lr=disc_lr, betas=(0.9, 0.999))



    for g in gen_optimizer.param_groups:
        g.setdefault('initial_lr', g['lr'])

    for d in disc_optimizer.param_groups:
        d.setdefault('initial_lr', d['lr'])


    sched_G = CosineAnnealingLR(gen_optimizer,
                                T_max=80,
                                last_epoch=gen_epoch,
                                eta_min=1e-6)
    
    sched_D = CosineAnnealingLR(disc_optimizer,
                            T_max=60,
                            last_epoch=disc_epoch,
                            eta_min=1e-6)
    
    sigma_sched = ScheduledSigma(sigma_start=0.6, sigma_end=0.0, T_max=900, mode='cosine', epoch=disc_epoch)

    for m in disc_model.modules():
        if isinstance(m, ScheduledDropout):
            m.setEpoch(disc_epoch)

    if False:
        if gen_state and args.train != "": #If we loaded checkpoint and are training, load any momentum from the optimizer
                gen_optimizer.load_state_dict(gen_state['optimizer_state_dict'])

        if disc_state and args.train != "": #If we loaded checkpoint and are training, load any momentum from the optimizer
                disc_optimizer.load_state_dict(disc_state['optimizer_state_dict'])


    #Assign the criterion for calculating loss
    gen_dist_criterion = nn.L1Loss()
    gen_dir_criterion = nn.CosineEmbeddingLoss()
    disc_criterion = nn.BCEWithLogitsLoss()

    #Set the runMode variable based on the arguments when launching
    if (True):
        runMode = TRAINING
    else:
        runMode = RUNNING_INFERENCE

    inferenceSteps = 300
    inferenceFrequency = 50
    inferenceCounter = 0

    name = "./outputs/10_24Simulation"
    id = "000"

    if False:
        runAccuracyTest(gen_model, device)
        return


    for i in range(1, epochs):
        printAndLog("Starting Epoch " + str(gen_epoch + i) + " for the generator.")
        printAndLog("Starting Epoch " + str(disc_epoch + i) + " for the discriminator.")


        if runMode == TRAINING:
            train(gen_model, disc_model, loader, gen_optimizer, disc_optimizer, gen_dist_criterion, gen_dir_criterion, disc_criterion, device, i)
                        # Save weights for later use
            torch.save({
                'epoch': gen_epoch + i,
                'model_state_dict': gen_model.state_dict(),
                'optimizer_state_dict': gen_optimizer.state_dict(),
            }, "gen_check/unet3d_vae_checkpoint" + str(gen_epoch+i) + ".pth")

            torch.save({
                'epoch': gen_epoch + i,
                'model_state_dict': gen_model.state_dict(),
                'optimizer_state_dict': gen_optimizer.state_dict(),
            }, "gen_check/latest.pth")

            printAndLog("Saved generator checkpoint at " + os.path.abspath("gen_check/unet3d_vae_checkpoint" + str(gen_epoch+i) + ".pth"))

            sched_D.step()
            sched_G.step()

            torch.save({
                'epoch': disc_epoch + i,
                'model_state_dict': disc_model.state_dict(),
                'optimizer_state_dict': disc_optimizer.state_dict(),
            }, "disc_check/cnn3d_vae_checkpoint" + str(disc_epoch+i) + ".pth")

            torch.save({
                'epoch': disc_epoch + i,
                'model_state_dict': disc_model.state_dict(),
                'optimizer_state_dict': disc_optimizer.state_dict(),
            }, "disc_check/latest.pth")

            printAndLog("Saved discriminator checkpoint at " + os.path.abspath("disc_check/cnn3d_vae_checkpoint" + str(disc_epoch+i) + ".pth"))

        #elif runMode == 2:
            #evaluate(gen_model, disc_model, loader, gen_optimizer, disc_optimizer, gen_criterion, disc_criterion, device)
        elif runMode == RUNNING_INFERENCE:
            inferenceCounter = inferenceCounter + 1
            if (inferenceCounter * inferenceFrequency) > inferenceSteps:
                break
            volumes, gt_instances = getInferenceData("D:\\runs\\runs\\run_300\\outputs_05 - Copy\\output000.piff")
            pred_label, all_matches, pred_ids = runinference(gen_model, volumes, gt_instances, inferenceCounter * inferenceFrequency, device)
            name = "./sim/6_25Simulation"
            id = f"{(int(id) + 1):03d}"
            #buildPiff(pred_label, all_matches, pred_ids, "D:\\runs\\runs\\run_300\\outputs_05 - Copy\\AIoutput" + f"{(inferenceCounter * inferenceFrequency):03d}" + ".piff")
            buildPiffNoMatch(pred_label, "D:\\runs\\runs\\run_300\\outputs_05 - Copy\\AIoutput" + f"{(inferenceCounter * inferenceFrequency):03d}" + ".piff")

        elif runMode == TESTING_POST_PROCESSING:
            name = "10_24Simulation"
            id = "000"
            volumes, gt_instances = getInferenceData(name, id)
            pred_label, all_matches, pred_ids = testProcessing( volumes, gt_instances, device)
            name = "10_24SimulationV2"
            id = f"{(int(id) + 1):03d}"
            buildPiff(pred_label, all_matches, pred_ids, name, id)
            


        if printGpuCheck:
            print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
            print("Reserved: ", torch.cuda.memory_reserved()  / 1024**2, "MB")
            # or for the full dump:
            print(torch.cuda.memory_summary())

        
        #print("^^^^^^^^^^^^^^ Epoch ", epoch, "^^^^^^^^^^^")
        #print("------------------------------")




if __name__ == "__main__":
   main()
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
from Utils import parse_voxel_file, parse_voxel_file_for_ID_matching, parse_voxel_file_for_distance
from Utils import get_voxel_center, voxel_points_to_self, quiver_slice_zyx
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

def plotTensor(tensor, batchIdx=0):
    probs = tensor[:, 3:, ...]                 # [B, K, D, H, W] (or [B, K, H, W])

    idx = probs.argmax(dim=1, keepdim=True)         # [B, 1, ...]
    one_hot = torch.zeros_like(probs).scatter_(1, idx, 1)

    final = one_hot[:, 0:2, ...].sum(dim=1, keepdim=True).clamp_max(1) 

    restored = final.repeat(1, 3, 1, 1, 1)

    quiver_slice_zyx((tensor[:, :3, ...]*restored)[batchIdx],  axis='y', index=96, stride=1)


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

def train(gen_model, disc_model, dataloader, gen_optimizer, disc_optimizer, gen_dist_criterion, gen_dir_criterion, disc_criterion, device, epochNumber):
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

    real_weight = 1
    fake_weight = 1

    adv_weight = 0.2
    gen_prob_weight = 0.4
    gen_dir_weight = 1

    weight_sum = adv_weight + gen_prob_weight + gen_dir_weight

    adv_weight = adv_weight/weight_sum
    gen_prob_weight = gen_prob_weight/weight_sum
    gen_dir_weight = gen_dir_weight/weight_sum

    total = 0

    skipThisGenBackProp = False
    skipThisDiscBackProp = False

    for volumes, targets, steps, path1, path2 in dataloader:
        volumes = volumes.to(device)
        targets = targets.to(device)


        assert torch.isfinite(volumes).all()
        assert torch.isfinite(targets).all()

        if skipNextDiscBackProp:
            skipThisDiscBackProp = True

        if skipNextGenBackProp:
            skipThisGenBackProp = True


        #B = volumes.shape[0]
        #z = torch.randn(B, noise_dim, device=device) * 0.1

        #Generate our predicted values
        gen_optimizer.zero_grad()
        #with autocast():

        #if (not voxel_points_to_self(targets[0], 100,100,100)):
            #raise Exception
        
        B, _, D, H, W = volumes.shape


        gen_outputs = gen_model(volumes, steps[0].item())
                

        #directions = (gen_outputs[:, :3, ...] / gen_outputs[:, :3, ...].norm(dim=1, keepdim=True).clamp_min(1e-8)).clone()
        directions = gen_outputs[:, :3, ...]

        probabilities = gen_outputs[:, 3:, ...]   

        target_directions = targets[:, :3, ...]

        valid = ((targets[:,3,...] == 1) | (targets[:,4,...] == 1))
        
        target_probabilities = targets[:, 3:, ...]


        #quiver_slice_zyx(volumes[0],  axis='y', index=98, stride=1, arrowScale=10.0, exclude_boundary_target=False)
        
        #gen_dir_loss = (1 - F.cosine_similarity(directions, target_directions, dim=1))[valid].mean()

        
        
        
        
        #mask = valid.float()  # [B,D,H,W]

        #per_comp  = F.smooth_l1_loss(directions, target_directions, reduction='none')   # [B,3,D,H,W]
        #per_voxel = per_comp.mean(dim=1)                            # [B,D,H,W]

        #num = (per_voxel * mask).sum(dim=(1,2,3))
        #den = mask.sum(dim=(1,2,3)).clamp_min(1)
        #per_sample = num / den

        #gen_dir_loss = per_sample.mean()




        eps = 1e-8


        # Normalize the direction channels
        p = torch.tanh(directions.clone())   # bound raw logits
        g = target_directions.detach()       # no grad through targets

        # Compute norms safely
        p_norm = torch.linalg.norm(p, dim=1, keepdim=True).clamp_min(eps)
        g_norm = torch.linalg.norm(g, dim=1, keepdim=True).clamp_min(eps)

        # Replace *first three* channels with normalized unit vectors
        gen_outputs[:, :3, ...] = p / p_norm
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


        gen_prob_loss = F.cross_entropy(probabilities, target_probabilities)
        
    
        #target_idx = target_probabilities.argmax(dim=1).long()
        #gen_prob_loss = F.cross_entropy(probabilities, target_idx)

        target_probabilities = blur_targets(targets[:, 3:, ...], kernel_size=3, sigma=sigma_sched.value)
        
        
        disc_optimizer.zero_grad()
        
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


        if not skipThisDiscBackProp:
            print("Doing disc backprop")
            set_requires_grad(gen_model, False)
            assert torch.isfinite(fake_loss).all()
            assert torch.isfinite(real_loss).all()
            #scaler.scale((fake_loss * fake_weight) + (real_loss*real_weight)).backward()
            #scaler.step(disc_optimizer)
            #scaler.update()
            disc_optimizer.zero_grad()
            ((fake_loss * fake_weight) + (real_loss * real_weight)).backward()
            disc_optimizer.step()
            set_requires_grad(gen_model, True)
        else:
            print("Skipping disc backprop")

        disc_outputs2 = disc_model(fake_output)

        adv_loss = disc_criterion(disc_outputs2, torch.ones_like(disc_outputs2))

        #finalize generated loss
        if not skipThisGenBackProp:
            set_requires_grad(disc_model, False)
            assert torch.isfinite(gen_dir_loss).all()
            assert torch.isfinite(gen_prob_loss).all()
            assert torch.isfinite(adv_loss).all()
            #scaler.scale((gen_dir_loss*gen_dir_weight) + (gen_prob_loss*gen_prob_weight) + (adv_loss*adv_weight)).backward()
            #scaler.step(gen_optimizer)
            #scaler.update()
            gen_optimizer.zero_grad()
            ((gen_dir_loss*gen_dir_weight) + (gen_prob_loss*gen_prob_weight) + (adv_loss*adv_weight)).backward()
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

        if skipThisDiscBackProp:
            skipThisDiscBackProp = False
            skipNextDiscBackProp = False
        elif ((real_loss + fake_loss).item() /2) < 0.5:
            #skipNextDiscBackProp = True
            print("(real_loss + fake_loss) /2) < 0.5")
        else:
            skipNextDiscBackProp = False

        if skipThisGenBackProp:
            skipThisGenBackProp = False
            skipNextGenBackProp = False
        elif (fake_loss.item() /2) > 0.8:
            #skipNextGenBackProp = True
            print("fake_loss > 0.8")
        else:
            skipNextGenBackProp = False

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
                    output = f"outputs_{(outputNumber+1):02d}"
                    stepNumber=(steps+1)*50
                    _,_ = parse_voxel_file_for_distance(paths[idx] + "\\" + output + f"\\output{stepNumber:03d}.piff", device=device, parallel=parallel)
                    
    

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
    parser.add_argument('--batchSize', type=int, default=2, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of Epochs to run')
    parser.add_argument('--gen_lr', type=float, default=4e-4, help='Learning rate for the Generator')
    parser.add_argument('--disc_lr', type=float, default=1e-4, help='Learning rate for the Discriminator')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use')
    parser.add_argument('--shuffle', type=bool, default=True, help='Whether to Shuffle the data')
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
        persistent_workers=args.persistent_workers
        
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
        gen_model.load_state_dict(gen_state['model_state_dict']) #Load the weights and biases

    if disc_state: #We loaded a checkpoint. (We wouldn't if it was the first run)
        disc_epoch = disc_state['epoch'] #Load the checkpoint the discriminator was on
        
        disc_model.load_state_dict(disc_state['model_state_dict']) #Load the weights and biases


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
#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from Utils import oneHotToDistance,smoothDistance, oneHotToDistance_fast,drop_nearby_by_count
import torch
from time import perf_counter
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import distance_transform_edt, zoom
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from scipy.optimize import linear_sum_assignment
from Utils import buildPiff, parse_voxel_file_for_distance,quiver_slice_zyx
from Constants import MAX_VOXEL_DIM

CENTER_VALUE = 40  # overlay value for centers




# -------------------------------



def render_yz_slice(base_vol, centers_zyx, x_s):
    """
    base_vol: (D,H,W) uint8 with 0=empty, 1=solid
    centers_zyx: list of (z,y,x) centers
    x_s: integer X slice index
    Returns a 2D image (H,D) with centers overlayed as CENTER_VALUE.
    """
    D, H, W = base_vol.shape
    img = base_vol[:, :, x_s].T.copy()  # (H, D)

    for (z, y, x) in centers_zyx:
        if x == x_s and 0 <= y < H and 0 <= z < D:
            img[y, z] = CENTER_VALUE

    return img




def _to_numpy_3dhw(t):
    """Accepts torch or numpy; returns np.ndarray float32 of shape (3,D,H,W) with positive strides."""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    t = np.asarray(t)
    assert t.ndim == 4 and t.shape[0] == 3, f"expected (3,D,H,W), got {t.shape}"
    return t.astype(np.float32, copy=True)  # copy=True avoids negative-stride issues

def signed_offsets_to_written_center(coords, spacing=(1.0, 1.0, 1.0)):
    """
    coords:  (3, D, H, W) with per-voxel written center coordinates [Xc, Yc, Zc] in *voxel indices*.
             i.e., coords[0] = Xc, coords[1] = Yc, coords[2] = Zc
    spacing: (dz, dy, dx) voxel spacing if anisotropic (used only if you also want physical offsets).

    Returns:
      offsets_vox: (3, D, H, W) = [dx_vox, dy_vox, dz_vox], signed offsets in *voxels*.
                   Example: one voxel left of center → dx_vox = -1; right → +1.
    """
    C = np.asarray(coords, dtype=np.float32)
    assert C.ndim == 4 and C.shape[0] == 3, "coords must be (3, D, H, W)"
    _, D, H, W = C.shape

    # index grids (voxel coordinates)
    Z, Y, X = np.meshgrid(np.arange(D, dtype=np.float32),
                          np.arange(H, dtype=np.float32),
                          np.arange(W, dtype=np.float32),
                          indexing='ij')

    # coords: [Xc, Yc, Zc] stored as voxel indices
    Xc = C[0]
    Yc = C[1]
    Zc = C[2]

    # signed offsets in *voxels*
    #dx = Xc - X
    #dy = Yc - Y
    #dz = Zc - Z

    dx_vox = X - Xc   # left = -1, right = +1
    dy_vox = Y - Yc   # up    = -1, down  = +1  (depending on your convention)
    dz_vox = Z - Zc   # toward -z = -1, toward +z = +1

    offsets_vox = np.stack([dx_vox, dy_vox, dz_vox], axis=0).astype(np.float32, copy=False)
    return offsets_vox




# Visualize direction–distance field on a 2D slice (for zyx layout)
# dist: torch.Tensor or np.ndarray with shape (4, Z, Y, X)
# axis: which constant-axis plane to show: 'z' -> XY plane, 'y' -> XZ, 'x' -> YZ
# index: slice index along that axis
# stride: decimation factor to avoid arrow clutter
# project: 'disp' -> use displacement r*u (recommended); 'unit' -> use unit u only
def quiver_slice_zyx2(dist, axis='z', index=0, stride=2, project='disp'):
    import numpy as np
    import matplotlib.pyplot as plt

    try:
        import torch
        is_torch = isinstance(dist, torch.Tensor)
    except Exception:
        is_torch = False

    A = dist.detach().float().cpu().numpy() if is_torch else np.asarray(dist, np.float32)
    assert A.ndim == 4 and A.shape[0] == 4, "expect (4, Z, Y, X)"

    ux, uy, uz, r = A[0], A[1], A[2], A[3]
    invalid = (r < 0)

    # displacement components (X,Y,Z) per voxel
    dx, dy, dz = ux * r, uy * r, uz * r
    # or unit components if requested
    if project == 'unit':
        dx, dy, dz = ux, uy, uz

    if axis == 'z':
        # XY plane at fixed Z=index
        U, V = dx[index, :, :], dy[index, :, :]
        mask = ~invalid[index, :, :]
        Xdim = A.shape[3]; Ydim = A.shape[2]
        Xg, Yg = np.meshgrid(np.arange(Xdim), np.arange(Ydim))
        xlabel, ylabel = "X", "Y"
    elif axis == 'y':
        # XZ plane at fixed Y=index (horizontal=X, vertical=Z)
        U, V = dx[:, index, :], dz[:, index, :]
        mask = ~invalid[:, index, :]
        Xdim = A.shape[3]; Zdim = A.shape[1]
        Xg, Yg = np.meshgrid(np.arange(Xdim), np.arange(Zdim))
        xlabel, ylabel = "X", "Z"
    elif axis == 'x':
        # YZ plane at fixed X=index (horizontal=Y, vertical=Z)
        U, V = dy[:, :, index], dz[:, :, index]
        mask = ~invalid[:, :, index]
        Ydim = A.shape[2]; Zdim = A.shape[1]
        Xg, Yg = np.meshgrid(np.arange(Ydim), np.arange(Zdim))
        xlabel, ylabel = "Y", "Z"
    else:
        raise ValueError("axis must be one of 'x','y','z'")

    # decimate for readability
    sl = (slice(None, None, stride), slice(None, None, stride))
    U, V, Xg, Yg, mask = U[sl], V[sl], Xg[sl], Yg[sl], mask[sl]

    # zero-out invalids (or you can drop them)
    U = np.where(mask, U, 0.0)
    V = np.where(mask, V, 0.0)

    mag2d = np.hypot(U, V)  # color by in-plane magnitude

    plt.figure(figsize=(6, 6))
    plt.quiver(Xg, Yg, U, V, mag2d, angles='xy', scale_units='xy', scale=1, pivot='tail')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(f"Quiver ({axis}-slice @ {index}) [{project}]")
    # If you want array-style top-left origin, uncomment:
    # plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def buildPiffOld(NewDistanceTensor: torch.Tensor,
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
        ux = T[0, 0].float().cpu()
        uy = T[0, 1].float().cpu()
        uz = T[0, 2].float().cpu()
        r  = T[0, 3].float().cpu()
    elif T.ndim == 4 and T.shape[0] == 4:     # [4,Z,Y,X]
        ux = T[0].float().cpu()
        uy = T[1].float().cpu()
        uz = T[2].float().cpu()
        r  = T[3].float().cpu()
    else:
        raise ValueError("Expected NewDistanceTensor as (B,4,Z,Y,X) or (4,Z,Y,X) with [ux,uy,uz,r]")

    Z, Y, X = r.shape

    # keep EVERY voxel with r >= 0
    valid = (r >= 0)
    if not valid.any():
        open(path, "w").close()
        return 0

    # indices of all valid voxels (Z,Y,X order)
    z_idx, y_idx, x_idx = torch.nonzero(valid, as_tuple=True)
    M = z_idx.numel()

    # Original centers (ID,Z,Y,X) on CPU
    if OriginalCenters.numel() == 0:
        raise ValueError("OriginalCenters is empty")
    orig_ids = OriginalCenters[:, 0].long().cpu()            # (N,)
    orig_zyx = OriginalCenters[:, 1:4].float().cpu()         # (N,3) Z,Y,X

    # start fresh file
    open(path, "w").close()
    total = 0
    start = 0
    while start < M:
        end = min(start + chunk_voxels, M)

        # slice this batch's voxel coords
        zi = z_idx[start:end]
        yi = y_idx[start:end]
        xi = x_idx[start:end]

        # per-voxel fields (indexing avoids any ordering mismatch)
        ux_f = ux[zi, yi, xi]
        uy_f = uy[zi, yi, xi]
        uz_f = uz[zi, yi, xi]
        r_f  =  r[zi, yi, xi]

        x0 = xi.float()
        y0 = yi.float()
        z0 = zi.float()

        # continuous pointed center (strict X,Y,Z)
        cx = x0 + ux_f * r_f
        cy = y0 + uy_f * r_f
        cz = z0 + uz_f * r_f

        # match pointed centers DIRECTLY to OriginalCenters (Z,Y,X)
        P = torch.stack([cz, cy, cx], dim=1)                 # (B,3) Z,Y,X
        d = torch.cdist(P, orig_zyx, p=2)                    # (B,N)
        nn = torch.argmin(d, dim=1)                          # (B,)
        cid = orig_ids[nn]                                   # (B,)

        # write rows
        cids = cid.tolist()
        celltype = ["Wall" if int(c) == int(wallID) else "Body" for c in cids]
        df = pd.DataFrame({
            "CellID": cids,
            "CellType": celltype,
            "x1": xi.tolist(), "x2": xi.tolist(),
            "y1": yi.tolist(), "y2": yi.tolist(),
            "z1": zi.tolist(), "z2": zi.tolist(),
        }, columns=["CellID","CellType","x1","x2","y1","y2","z1","z2"])

        
        df.to_csv(path, sep=' ', index=False, header=False, mode='a')

        total += (end - start)
        start = end

    return df


import torch

def drop_nearby_by_count2(result: torch.Tensor, radius: float = 3.0, metric: str = "euclidean"):
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
        if suppressed[idx]:
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

def visualize_distances(distances, z):
    slice_2d = distances[z].cpu().numpy()

    # Mask out values >= 125
    masked = slice_2d.copy()
    masked[masked >= 125] = float("nan")   # NaNs won’t be shown in imshow
    masked[masked == float("inf")] = float("nan")

    plt.imshow(masked, cmap="viridis")
    plt.colorbar(label="distance (<125 only)")
    plt.title("Slice at index 100, values <125")
    plt.show()



import torch
import torch.nn.functional as F

def _shift_inf(t, dz, dy, dx):
    Z, Y, X = t.shape
    pad = (max(dx,0), max(-dx,0), max(dy,0), max(-dy,0), max(dz,0), max(-dz,0))
    tp = F.pad(t, pad, value=float('inf'))
    z0 = max(-dz,0); y0 = max(-dy,0); x0 = max(-dx,0)
    return tp[z0:z0+Z, y0:y0+Y, x0:x0+X]

def geodesic_chamfer3d(mask: torch.Tensor, seed_zyx, sweeps: int = 2, weights=(10.0, 14.0, 17.0)):
    """
    mask: (Z,Y,X) bool tensor (True = allowed)
    seed_zyx: (z,y,x) inside mask
    sweeps: number of forward/backward sweep pairs
    weights: (axial, face-diag, space-diag) chamfer weights
    returns: (Z,Y,X) float distances (inf outside mask), scaled so axial step ≈ 1.0
    """
    axial, face, space = weights
    device = mask.device
    Z, Y, X = mask.shape

    inf = torch.tensor(float('inf'), device=device)
    dist = torch.full((Z, Y, X), float('inf'), device=device)
    z0, y0, x0 = map(int, seed_zyx)
    dist[z0, y0, x0] = 0.0

    # 13-neighbor stencils per sweep (causal)
    fwd = [(-1,-1,-1,space), (-1,-1, 0,face), (-1,-1, 1,space),
           (-1, 0,-1,face),  (-1, 0, 0,axial),(-1, 0, 1,face),
           (-1, 1,-1,space), (-1, 1, 0,face), (-1, 1, 1,space),
           ( 0,-1,-1,face),  ( 0,-1, 0,axial),( 0,-1, 1,face),
           ( 0, 0,-1,axial)]
    bwd = [( 1, 1, 1,space), ( 1, 1, 0,face), ( 1, 1,-1,space),
           ( 1, 0, 1,face),  ( 1, 0, 0,axial),( 1, 0,-1,face),
           ( 1,-1, 1,space), ( 1,-1, 0,face), ( 1,-1,-1,space),
           ( 0, 1, 1,face),  ( 0, 1, 0,axial),( 0, 1,-1,face),
           ( 0, 0, 1,axial)]

    for _ in range(sweeps):
        for dz,dy,dx,w in fwd:
            cand = _shift_inf(dist, dz,dy,dx) + w
            dist = torch.where(mask, torch.minimum(dist, cand), inf)
        for dz,dy,dx,w in bwd:
            cand = _shift_inf(dist, dz,dy,dx) + w
            dist = torch.where(mask, torch.minimum(dist, cand), inf)

    # convert back to ~Euclidean units
    return dist / axial

def oneHotToDistance_chamfer(source, centers, sweeps=2):
    """
    source: (D,H,W) labels (0=background, 1..K), or (1,D,H,W)
    centers: list of (ID, z, y, x) per instance; ID must match label in `source`.
    returns: finalDistances (D,H,W), channelDistances (K,D,H,W)
    """
    lab = source[0] if source.ndim == 4 else source
    D,H,W = lab.shape
    device = lab.device
    K = len(centers)
    channelDistances = torch.full((K, D, H, W), float('inf'), device=device)

    finalDistances = torch.zeros((D, H, W), device=device)
    for i, (ID, zc, yc, xc) in enumerate(centers):
        ID = int(ID)
        m = (lab == ID)

        nz = m.nonzero(as_tuple=False)
        if nz.numel() == 0:
            continue
        zmin,ymin,xmin = nz.min(dim=0).values.tolist()
        zmax,ymax,xmax = (nz.max(dim=0).values + 1).tolist()

        m_roi = m[zmin:zmax, ymin:ymax, xmin:xmax]
        dz, dy, dx = int(zc)-zmin, int(yc)-ymin, int(xc)-xmin

        dist_roi = geodesic_chamfer3d(m_roi, (dz,dy,dx), sweeps=sweeps)
        channelDistances[i, zmin:zmax, ymin:ymax, xmin:xmax] = dist_roi

        finalDistances[zmin:zmax, ymin:ymax, xmin:xmax] += torch.where(
            torch.isfinite(dist_roi), dist_roi, torch.tensor(0.0, device=device)
        )

    return finalDistances, channelDistances


import torch
import torch.nn.functional as F

@torch.no_grad()
def splat_flow_nearest_from(flow_dirs: torch.Tensor, flow_vals: torch.Tensor):
    """
    Nearest-neighbor push using TWO tensors:
      - flow_dirs gives the 3D displacement (dz,dy,dx)
      - flow_vals provides the 3-vector to ADD at the destination voxel
    Shapes: (3,D,H,W) or (N,3,D,H,W). Returns accumulated field same shape as flow_vals.
    """

    mask = (torch.isnan(flow_dirs) | torch.isnan(flow_vals))

    flow_dirs[mask] = 0
    flow_vals[mask] = 0

    assert flow_vals.shape == flow_dirs.shape, "flow_vals and flow_dirs must have identical shapes"
    has_batch = (flow_dirs.ndim == 5)
    if not has_batch:
        flow_vals = flow_vals.unsqueeze(0); flow_dirs = flow_dirs.unsqueeze(0)

    N,C,D,H,W = flow_dirs.shape
    assert C == 3

    dz, dy, dx = flow_dirs[:,0], flow_dirs[:,1], flow_dirs[:,2]
    mag = torch.ones_like(dx)
    valid = torch.isfinite(mag) & (mag > 0)

    # match your working behavior: use given components as the step (no normalization)
    uz = torch.where(valid, dz, torch.zeros_like(dz))
    uy = torch.where(valid, dy, torch.zeros_like(dy))
    ux = torch.where(valid, dx, torch.zeros_like(dx))

    # base grid
    z = torch.arange(D, device=flow_dirs.device, dtype=torch.float32)
    y = torch.arange(H, device=flow_dirs.device, dtype=torch.float32)
    x = torch.arange(W, device=flow_dirs.device, dtype=torch.float32)
    Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')
    Z = Z.unsqueeze(0).expand(N,-1,-1,-1)
    Y = Y.unsqueeze(0).expand(N,-1,-1,-1)
    X = X.unsqueeze(0).expand(N,-1,-1,-1)

    # target coords (nearest), invalid sources stay put
    TZ = torch.where(valid, Z + uz, Z).round().clamp_(0, D-1).to(torch.long)
    TY = torch.where(valid, Y + uy, Y).round().clamp_(0, H-1).to(torch.long)
    TX = torch.where(valid, X + ux, X).round().clamp_(0, W-1).to(torch.long)

    # flat destination index
    dest = (TZ*H*W + TY*W + TX).view(N, -1)        # (N,P)
    idx_exp = dest.unsqueeze(1).expand(N, C, -1)   # (N,3,P)

    # gather VALUES from flow_vals at destination, add into destination
    vals_flat = flow_vals.view(N, C, -1)
    dest_vecs = torch.gather(vals_flat, 2, idx_exp)

    w = valid.view(N, 1, -1).to(flow_vals.dtype)   # only valid sources contribute
    contrib = dest_vecs * w

    out = torch.zeros_like(vals_flat)
    out.scatter_add_(2, idx_exp, contrib)
    out = out.view(N, C, D, H, W)
    if not has_batch:
        out = out.squeeze(0)
    out[mask] = torch.nan
    return out


@torch.no_grad()
def splat_flow_nearest(flow: torch.Tensor):
    """
    For each voxel, jump 1 voxel along its 3D flow direction (nearest neighbor),
    then ADD the vector at the destination voxel into that destination.
    flow: (3,D,H,W) or (N,3,D,H,W); channels=(dz,dy,dx). inf/NaN or zero -> ignored.
    Returns: accumulated field, same shape as flow.
    """
    has_batch = (flow.ndim == 5)
    if not has_batch:
        flow = flow.unsqueeze(0)  # (1,3,D,H,W)
    N,C,D,H,W = flow.shape
    assert C == 3

    dz, dy, dx = flow[:,0], flow[:,1], flow[:,2]
    mag = torch.sqrt(dz*dz + dy*dy + dx*dx)
    valid = torch.isfinite(mag) & (mag > 0)

    # unit directions only where valid
    inv_mag = torch.zeros_like(mag)
    inv_mag[valid] = 1.0 #/ (mag[valid] + 1e-12)
    uz = dz * inv_mag
    uy = dy * inv_mag
    ux = dx * inv_mag

    # base grid
    z = torch.arange(D, device=flow.device, dtype=torch.float32)
    y = torch.arange(H, device=flow.device, dtype=torch.float32)
    x = torch.arange(W, device=flow.device, dtype=torch.float32)
    Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')  # (D,H,W)
    Z = Z.unsqueeze(0).expand(N,-1,-1,-1)
    Y = Y.unsqueeze(0).expand(N,-1,-1,-1)
    X = X.unsqueeze(0).expand(N,-1,-1,-1)

    # target coords (nearest)
    TZ = Z + uz
    TY = Y + uy
    TX = X + ux

    # for invalid sources, force self-index (no move)
    TZ = torch.where(valid, TZ, Z)
    TY = torch.where(valid, TY, Y)
    TX = torch.where(valid, TX, X)

    # round to nearest and clamp; convert to long safely
    TZ = TZ.round().clamp_(0, D-1).to(torch.long)
    TY = TY.round().clamp_(0, H-1).to(torch.long)
    TX = TX.round().clamp_(0, W-1).to(torch.long)

    # flat destination index
    dest = (TZ*H*W + TY*W + TX).view(N, -1)  # (N,P)

    flow_flat = flow.view(N, C, -1)          # values field (source & destination live here)
    idx_exp   = dest.unsqueeze(1).expand(N, C, -1)

    # gather destination vectors; zero out invalid sources
    dest_vecs = torch.gather(flow_flat, 2, idx_exp)         # (N,3,P)
    w = valid.view(N, 1, -1).to(flow.dtype)
    contrib = dest_vecs * w

    # scatter-add into output
    out = torch.zeros_like(flow_flat)
    out.scatter_add_(2, idx_exp, contrib)
    out = out.view(N, C, D, H, W)
    return out.squeeze(0) if not has_batch else out


import torch

import torch

import torch

import torch
from typing import Optional, Tuple

import torch
from typing import Optional, Tuple

import torch
from typing import Optional, Tuple

import torch
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from typing import Optional, Tuple

import torch
from typing import Optional, Tuple

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

import torch
from typing import Optional, Tuple

import torch
from typing import Optional, Tuple

@torch.no_grad()
def remove_vectors_pointing_to_nan(
    flow_vals: torch.Tensor,   # (3,D,H,W) or (N,3,D,H,W) — vectors
    flow_dirs: torch.Tensor,   # (3,D,H,W) or (N,3,D,H,W) — displacement directions
    fill_value: float = 0.0    # value to write for removed vectors
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Zeroes (or fills) any vector whose rounded+clamped destination lands on a voxel
    where the destination vector has any NaN component.

    Returns:
      flow_out : same shape as flow_vals, with offending vectors set to fill_value
      mask     : (D,H,W) or (N,D,H,W) bool mask True where a vector was removed
    """
    assert flow_vals.shape == flow_dirs.shape and flow_vals.dim() in (4,5)
    has_batch = (flow_vals.dim() == 5)
    if not has_batch:
        flow_vals = flow_vals.unsqueeze(0)
        flow_dirs = flow_dirs.unsqueeze(0)

    N, C, D, H, W = flow_vals.shape
    assert C == 3
    dev = flow_vals.device

    # Directions and validity
    dz, dy, dx = flow_dirs[:, 0], flow_dirs[:, 1], flow_dirs[:, 2]
    mag   = torch.sqrt(dz*dz + dy*dy + dx*dx)
    valid = torch.isfinite(mag) & (mag > 0)

    # Base grid
    z = torch.arange(D, device=dev, dtype=torch.float32)
    y = torch.arange(H, device=dev, dtype=torch.float32)
    x = torch.arange(W, device=dev, dtype=torch.float32)
    Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')
    Zb, Yb, Xb = Z.unsqueeze(0).expand(N,-1,-1,-1), Y.unsqueeze(0).expand(N,-1,-1,-1), X.unsqueeze(0).expand(N,-1,-1,-1)

    # Proposed destination (rounded BEFORE clamp); invalid -> self
    RZ = torch.where(valid, Zb + dz, Zb).round()
    RY = torch.where(valid, Yb + dy, Yb).round()
    RX = torch.where(valid, Xb + dx, Xb).round()

    TZ = RZ.clamp_(0, D-1).to(torch.long)
    TY = RY.clamp_(0, H-1).to(torch.long)
    TX = RX.clamp_(0, W-1).to(torch.long)

    # Destination linear index per voxel
    dest_flat = (TZ * (H*W) + TY * W + TX).view(N, -1)  # (N,P), P=D*H*W

    # NaN at destination? (any channel)
    dest_nan_any = torch.isnan(flow_vals).any(dim=1)      # (N,D,H,W)
    dest_nan_flat = dest_nan_any.view(N, -1)              # (N,P)
    points_to_nan_flat = dest_nan_flat.gather(1, dest_flat)  # (N,P) bool
    points_to_nan = points_to_nan_flat.view(N, D, H, W)      # (N,D,H,W)

    # Remove (fill) offending vectors
    flow_out = flow_vals.clone()
    mask_exp = points_to_nan.unsqueeze(1).expand_as(flow_out)  # (N,3,D,H,W)
    flow_out = flow_out.masked_fill(mask_exp, fill_value)

    # Squeeze batch dim if unbatched
    if not has_batch:
        flow_out = flow_out.squeeze(0)
        points_to_nan = points_to_nan.squeeze(0)

    return flow_out, points_to_nan


import torch
from typing import Tuple

@torch.no_grad()
def point_direct_to_centers(flow: torch.Tensor,
                            step_mode: str = "nearest",   # "nearest" or "sign"
                            max_iters: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a local flow (3,D,H,W) into a field pointing directly to its terminal voxel (center/sink).
    Returns:
      out  : (3,D,H,W) direct displacement to center
      dest : (D*H*W,) flat index of terminal voxel per position

    Behavior: if the step would land on a voxel whose flow contains NaN in any channel,
    treat that as terminal and do not step into it (stop before NaN).
    """
    assert flow.ndim == 4 and flow.shape[0] == 3, "flow must be (3,D,H,W)"
    device = flow.device
    _, D, H, W = flow.shape
    P = D * H * W

    dz, dy, dx = flow[0], flow[1], flow[2]
    mag   = torch.sqrt(dz*dz + dy*dy + dx*dx)
    valid = torch.isfinite(mag) & (mag > 0)

    # NaN masks
    nanMask   = torch.isnan(flow)          # (3,D,H,W)
    nan_any   = nanMask.any(dim=0)         # (D,H,W) True if any channel is NaN at that voxel

    # base grid
    z = torch.arange(D, device=device, dtype=torch.float32)
    y = torch.arange(H, device=device, dtype=torch.float32)
    x = torch.arange(W, device=device, dtype=torch.float32)
    Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')  # non-contiguous

    # choose step per mode
    if step_mode == "sign":
        sz, sy, sx = torch.sign(dz), torch.sign(dy), torch.sign(dx)
    elif step_mode == "nearest":
        sz, sy, sx = dz, dy, dx
    else:
        raise ValueError("step_mode must be 'nearest' or 'sign'")

    # destinations (nearest integer step, clamped; invalid -> self)
    RZ = torch.where(valid, (Z + sz).round(), Z)
    RY = torch.where(valid, (Y + sy).round(), Y)
    RX = torch.where(valid, (X + sx).round(), X)

    TZ = RZ.clamp_(0, D-1).to(torch.long)
    TY = RY.clamp_(0, H-1).to(torch.long)
    TX = RX.clamp_(0, W-1).to(torch.long)

    ZL, YL, XL = Z.to(torch.long), Y.to(torch.long), X.to(torch.long)

    # If the proposed destination has NaN flow (any channel), stop before stepping into it
    step_hits_nan = nan_any[TZ, TY, TX]    # (D,H,W) bool

    self_lin = (ZL * (H*W) + YL * W + XL).reshape(-1)             # (P,)
    prop_lin = (TZ * (H*W) + TY * W + TX).reshape(-1)             # (P,)
    dest     = torch.where(step_hits_nan.reshape(-1), self_lin, prop_lin)  # (P,)

    # pointer jumping: dest <- dest[dest]
    for _ in range(max_iters):
        new_dest = dest.gather(0, dest)
        if torch.equal(new_dest, dest):
            break
        dest = new_dest

    # unravel final destinations to coordinates
    zc = (dest // (H*W)).to(torch.long)
    rem = dest % (H*W)
    yc = (rem // W).to(torch.long)
    xc = (rem %  W).to(torch.long)

    # direct displacement to center: (zc - z, yc - y, xc - x)
    Zf = ZL.reshape(-1)
    Yf = YL.reshape(-1)
    Xf = XL.reshape(-1)

    dz_dir = (zc - Zf).to(torch.float32).reshape(D, H, W)
    dy_dir = (yc - Yf).to(torch.float32).reshape(D, H, W)
    dx_dir = (xc - Xf).to(torch.float32).reshape(D, H, W)

    out = torch.stack([dz_dir, dy_dir, dx_dir], dim=0).to(flow.dtype).to(device)
    out = torch.where(valid.unsqueeze(0), out, torch.zeros_like(out))  # keep invalids at 0

    # Keep original NaNs at their locations
    out[nanMask] = torch.nan
    return out, dest


import torch
import math

@torch.no_grad()
def point_direct_to_centers2(flow: torch.Tensor,
                            step_mode: str = "nearest",   # "nearest" or "sign"
                            max_iters: int | None = None):
    """
    Convert local flow (3,D,H,W) into a field pointing directly to its terminal voxel (center/sink).
    Returns:
      out  : (3,D,H,W) displacement-to-center (zc-z, yc-y, xc-x)
      dest : (D*H*W,) long, terminal flat index per voxel
    """
    assert flow.ndim == 4 and flow.shape[0] == 3, "flow must be (3,D,H,W)"
    device, dtype = flow.device, flow.dtype
    _, D, H, W = flow.shape
    P = D * H * W

    # Flattened components
    dz = flow[0].reshape(-1)
    dy = flow[1].reshape(-1)
    dx = flow[2].reshape(-1)

    # Valid direction mask
    mag = torch.sqrt(dz*dz + dy*dy + dx*dx)
    valid = torch.isfinite(mag) & (mag > 0)

    # Optional NaN mask to propagate back
    nanMask = torch.isnan(flow)

    # Base linear -> (z,y,x)
    idx = torch.arange(P, device=device, dtype=torch.long)
    z = idx // (H*W)
    rem = idx %  (H*W)
    y = rem // W
    x = rem %  W

    # Step per mode (integer step)
    if step_mode == "sign":
        sz = torch.sign(dz)
        sy = torch.sign(dy)
        sx = torch.sign(dx)
    elif step_mode == "nearest":
        sz = dz
        sy = dy
        sx = dx
    else:
        raise ValueError("step_mode must be 'nearest' or 'sign'")

    sz = sz.round().to(torch.long)
    sy = sy.round().to(torch.long)
    sx = sx.round().to(torch.long)

    # Destinations (invalid -> self)
    tz = torch.where(valid, (z + sz).clamp_(0, D-1), z)
    ty = torch.where(valid, (y + sy).clamp_(0, H-1), y)
    tx = torch.where(valid, (x + sx).clamp_(0, W-1), x)
    dest = (tz * (H*W) + ty * W + tx)  # (P,) long

    # Pointer jumping (path compression). A few log2 steps are enough.
    if max_iters is None:
        max_iters = int(math.ceil(math.log2(max(D, H, W)))) + 3
    for _ in range(max_iters):
        new_dest = dest.gather(0, dest)
        if torch.equal(new_dest, dest):
            break
        dest = new_dest

    # Unravel final dest -> (zc,yc,xc)
    zc = dest // (H*W)
    rem = dest %  (H*W)
    yc = rem // W
    xc = rem %  W

    # Displacement to center (float)
    dz_dir = (zc - z).to(torch.float32).reshape(D, H, W)
    dy_dir = (yc - y).to(torch.float32).reshape(D, H, W)
    dx_dir = (xc - x).to(torch.float32).reshape(D, H, W)
    out = torch.stack([dz_dir, dy_dir, dx_dir], dim=0).to(dtype=dtype, device=device)

    # Zero where original flow was invalid; re-instate NaNs where they were
    out = torch.where(valid.reshape(1, D, H, W), out, torch.zeros_like(out))
    out[nanMask] = torch.nan
    return out, dest



import torch
import torch.nn.functional as F

@torch.no_grad()
def point_to_centers_grid(flow: torch.Tensor,
                          step: float = 1.0,
                          max_steps: int = 4000,
                          tol: float = 1e-3,
                          align_corners: bool = True):
    """
    Follow flow with trilinear interpolation (grid_sample) until stagnation.
      flow: (3,D,H,W)  (dz,dy,dx) in voxel units; NaN/Inf treated as zero.
      step: step size in voxels per iteration (e.g., 1.0 or 0.5).
      tol : stop when |dir| < tol.
    Returns:
      out_disp: (3,D,H,W) displacement-to-center = (zc-z, yc-y, xc-x)
      term_xyz: (3,D,H,W) float absolute terminal coords (z,y,x)
    """
    nanMask = (torch.isnan(flow))
    assert flow.ndim == 4 and flow.shape[0] == 3
    dev, dt = flow.device, flow.dtype
    _, D, H, W = flow.shape

    # base coordinates
    z0 = torch.arange(D, device=dev, dtype=dt)
    y0 = torch.arange(H, device=dev, dtype=dt)
    x0 = torch.arange(W, device=dev, dtype=dt)
    Z0, Y0, X0 = torch.meshgrid(z0, y0, x0, indexing='ij')         # (D,H,W)

    # current positions start at voxel centers
    Z = Z0.clone(); Y = Y0.clone(); X = X0.clone()

    # prepack input for grid_sample: (N=1,C=3,D,H,W)
    inp = flow.unsqueeze(0)

    # helper: build normalized grid from absolute coords
    def make_grid(Z, Y, X):
        if align_corners:
            gn_x = (X * 2 / (W - 1)) - 1
            gn_y = (Y * 2 / (H - 1)) - 1
            gn_z = (Z * 2 / (D - 1)) - 1
        else:
            gn_x = ((X + 0.5) * 2 / W) - 1
            gn_y = ((Y + 0.5) * 2 / H) - 1
            gn_z = ((Z + 0.5) * 2 / D) - 1
        return torch.stack([gn_x, gn_y, gn_z], dim=-1).unsqueeze(0)  # (1,D,H,W,3)

    # iterate: sample dir at current pos, step, clamp
    eps = 1e-4
    for _ in range(max_steps):
        grid = make_grid(Z, Y, X)
        dir_ = F.grid_sample(inp, grid, mode='bilinear',
                             align_corners=align_corners, padding_mode='border')[0]  # (3,D,H,W)
        # clean non-finite
        dir_ = torch.nan_to_num(dir_, nan=0.0, posinf=0.0, neginf=0.0)
        mag = torch.linalg.vector_norm(dir_, dim=0)  # (D,H,W)

        # stop mask
        moving = mag > tol
        if not moving.any():
            break

        # normalized step (avoid div-by-zero)
        u = torch.where(mag.unsqueeze(0) > 0, dir_ / (mag.unsqueeze(0) + 1e-12), torch.zeros_like(dir_))
        # advance only where moving
        Z = torch.where(moving, Z + u[0]*step, Z)
        Y = torch.where(moving, Y + u[1]*step, Y)
        X = torch.where(moving, X + u[2]*step, X)

        # keep in bounds (slightly inside to avoid hitting borders)
        Z.clamp_(0.0 + eps, D - 1.0 - eps)
        Y.clamp_(0.0 + eps, H - 1.0 - eps)
        X.clamp_(0.0 + eps, W - 1.0 - eps)

    # displacement to terminal positions
    dz = (Z - Z0).to(dt)
    dy = (Y - Y0).to(dt)
    dx = (X - X0).to(dt)
    out_disp = torch.stack([dz, dy, dx], dim=0)  # (3,D,H,W)
    term_xyz = torch.stack([Z, Y, X], dim=0)     # (3,D,H,W)

    out_disp[nanMask] = torch.nan
    return out_disp, term_xyz


import torch

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

import torch

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

import torch
from typing import Optional

@torch.no_grad()
def snap_vectors_to_nearest_non_nan(
    flow: torch.Tensor,
    search_radius: int = 1,
    keep_original_if_none: bool = True,
) -> torch.Tensor:
    """
    For each source voxel (z,y,x) with vector (dz,dy,dx), find a destination voxel near the rounded endpoint
    that has a finite vector (no NaNs/Infs). If the immediate rounded target is invalid, search a Chebyshev
    neighborhood of radius `search_radius` around it and pick the valid voxel closest (in index space) to the
    continuous endpoint. Source voxels with non-finite vectors are left unchanged.

    flow: (3, D, H, W)    components order: (dz, dy, dx)
    """
    assert flow.ndim == 4 and flow.shape[0] == 3, "flow must be (3,D,H,W)"
    _, D, H, W = flow.shape
    dev, dtype = flow.device, flow.dtype

    # Source validity: if source vector is non-finite, we leave it unchanged and we also avoid any NaN->long casts.
    src_finite = torch.isfinite(flow).all(dim=0)  # (D,H,W)

    # Base grid
    z = torch.arange(D, device=dev, dtype=dtype)
    y = torch.arange(H, device=dev, dtype=dtype)
    x = torch.arange(W, device=dev, dtype=dtype)
    Z, Y, X = torch.meshgrid(z, y, x, indexing='ij')  # (D,H,W)

    dz, dy, dx = flow[0], flow[1], flow[2]

    # Continuous endpoints
    PZ = Z + dz
    PY = Y + dy
    PX = X + dx

    # SAFETY: never cast NaN to long. For invalid sources, use the source index itself.
    PZs = torch.where(src_finite, PZ, Z)
    PYs = torch.where(src_finite, PY, Y)
    PXs = torch.where(src_finite, PX, X)

    # Rounded & clamped initial target (safe for invalid sources)
    TZ0 = torch.round(PZs).clamp(0, D - 1).to(torch.long)
    TY0 = torch.round(PYs).clamp(0, H - 1).to(torch.long)
    TX0 = torch.round(PXs).clamp(0, W - 1).to(torch.long)

    # Destination validity (any NaN/Inf at dest voxel -> invalid)
    dest_finite = torch.isfinite(flow).all(dim=0)     # (D,H,W)
    dest_finite_flat = dest_finite.view(-1)

    # Flat indices & initial validity (via gather; avoids risky advanced indexing)
    lin0 = (TZ0 * (H * W) + TY0 * W + TX0).view(-1)                # (P,)
    init_valid_flat = dest_finite_flat.gather(0, lin0).view(D, H, W)

    # Fast path: all initial targets valid -> snap directly
    if init_valid_flat.all().item():
        snapped_dz = TZ0.to(dtype) - Z
        snapped_dy = TY0.to(dtype) - Y
        snapped_dx = TX0.to(dtype) - X
        snapped = torch.stack([snapped_dz, snapped_dy, snapped_dx], dim=0)
        return torch.where(src_finite.unsqueeze(0), snapped, flow)

    # Initialize best = initial rounded target
    best_Z = TZ0.clone()
    best_Y = TY0.clone()
    best_X = TX0.clone()
    best_found = init_valid_flat.clone()

    # Use the *safe* endpoints (P*s) so distances are finite even for invalid sources
    best_dist2 = (PZs - TZ0.to(dtype))**2 + (PYs - TY0.to(dtype))**2 + (PXs - TX0.to(dtype))**2

    # Neighborhood offsets within Chebyshev radius R
    rng = range(-search_radius, search_radius + 1)
    offsets = [(oz, oy, ox) for oz in rng for oy in rng for ox in rng
               if not (oz == 0 and oy == 0 and ox == 0)]

    for oz, oy, ox in offsets:
        CZ = (TZ0 + oz).clamp(0, D - 1)
        CY = (TY0 + oy).clamp(0, H - 1)
        CX = (TX0 + ox).clamp(0, W - 1)
        clin = (CZ * (H * W) + CY * W + CX).view(-1)                 # (P,)
        cand_valid = dest_finite_flat.gather(0, clin).view(D, H, W)  # (D,H,W)

        # Distance from SAFE continuous endpoint to candidate index
        dist2 = (PZs - CZ.to(dtype))**2 + (PYs - CY.to(dtype))**2 + (PXs - CX.to(dtype))**2

        improve = cand_valid & (~best_found | (dist2 < best_dist2))
        best_Z = torch.where(improve, CZ, best_Z)
        best_Y = torch.where(improve, CY, best_Y)
        best_X = torch.where(improve, CX, best_X)
        best_dist2 = torch.where(improve, dist2, best_dist2)
        best_found = best_found | cand_valid

    # Build snapped displacement
    snapped_dz = best_Z.to(dtype) - Z
    snapped_dy = best_Y.to(dtype) - Y
    snapped_dx = best_X.to(dtype) - X
    snapped = torch.stack([snapped_dz, snapped_dy, snapped_dx], dim=0)

    # If some sources still couldn't find a valid dest in the window
    if not best_found.all().item():
        if keep_original_if_none:
            snapped = torch.where(best_found.unsqueeze(0), snapped, flow)
        # else: keep snapped (it points to the closest index we had)

    # Preserve invalid sources unchanged
    snapped = torch.where(src_finite.unsqueeze(0), snapped, flow)
    return snapped

import torch

@torch.no_grad()
def coord_match_and_unmatched(A: torch.Tensor, B: torch.Tensor):
    """
    A, B: (N,4)-ish tensors with columns [*, z, y, x].
    - Ignores column 0
    - Compares exact integer coordinates after rounding (ties-to-even)
    - Drops rows with non-finite z/y/x before matching

    Returns:
      matches_count: int = | set_A(z,y,x) ∩ set_B(z,y,x) |
      A_unmatched  : rows of A (finite z/y/x) whose coords don't appear in B
      B_unmatched  : rows of B (finite z/y/x) whose coords don't appear in A
    """
    assert A.ndim == 2 and B.ndim == 2 and A.size(1) >= 4 and B.size(1) >= 4

    # Keep only rows with all-finite coords
    A_keep = A[torch.isfinite(A[:, 1:4]).all(dim=1)]
    B_keep = B[torch.isfinite(B[:, 1:4]).all(dim=1)]

    if A_keep.numel() == 0 or B_keep.numel() == 0:
        return 0, A_keep, B_keep

    # Round coords to nearest int and hash (z,y,x) into a single int key
    Axyz = torch.round(A_keep[:, 1:4]).to(torch.long)
    Bxyz = torch.round(B_keep[:, 1:4]).to(torch.long)

    # Build a collision-free hash using max extents from both sets
    maxs = torch.stack([Axyz.max(0).values, Bxyz.max(0).values]).max(0).values + 1  # (3,)
    Mx = maxs[2]
    My = maxs[1]
    mul = torch.tensor([My * Mx, Mx, 1], device=A.device, dtype=torch.long)        # for (z,y,x)

    keysA = (Axyz * mul).sum(dim=1)
    keysB = (Bxyz * mul).sum(dim=1)

    # Set intersection size (unique coords)
    uA = torch.unique(keysA)
    uB = torch.unique(keysB)
    matched_set = uA[torch.isin(uA, uB)]
    matches_count = int(matched_set.numel())

    # Unmatched rows (by set membership)
    A_unmatched = A_keep[~torch.isin(keysA, matched_set)]
    B_unmatched = B_keep[~torch.isin(keysB, matched_set)]

    return matches_count, A_unmatched, B_unmatched

# Example:
# matches = count_coord_matches_1to3(aiCenters, centers)
# print(matches)



def vote_gap(rows: torch.Tensor, k: int) -> float:
    """
    rows: (N, 4) tensor with columns [votes, z, y, x]
    k:    1-based rank. k=32 returns votes[k]-votes[k+1] with votes sorted desc.

    Returns: float gap (votes_k - votes_kplus1).
    """
    assert rows.ndim == 2 and rows.size(1) >= 1, "expected (N,4) or (N,>=1)"
    votes = rows[:, 0]

    # keep only finite votes
    mask = torch.isfinite(votes)
    votes = votes[mask]
    if votes.numel() < (k + 1):
        raise ValueError(f"Need at least {k+1} finite rows; got {votes.numel()}")

    # sort descending and take k-th (1-based) and (k+1)-th
    vals, _ = torch.sort(votes, descending=True, stable=True)
    v_k   = vals[k-1]
    v_k1  = vals[k]
    return (v_k - v_k1).item()

def process_file(piffs, outdir = None, show = False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = perf_counter()
    DistanceTensor, _, Centers, WallID, wallMask = parse_voxel_file_for_distance(piffs[0], device)
    print(f"{perf_counter() - t0:.6f} s")


    DistanceTensor2, _, Centers2, _, _ = parse_voxel_file_for_distance(piffs[1])

    if False:
        quiver_slice_zyx(DistanceTensor.squeeze(0),  axis='y', index=98, stride=1)
        quiver_slice_zyx(DistanceTensor2.squeeze(0),  axis='y', index=98, stride=1)
    

    buildPiff(DistanceTensor2, Centers, WallID, ".\\VacuoleAutophagySim\\testCenterFinding\\aioutput050.piff")
    buildPiff(DistanceTensor2, Centers, WallID, ".\\VacuoleAutophagySim\\testCenterFinding\\aioutput050V2.piff")
    aiDistanceTensor, _, aiCenters, _,_ = parse_voxel_file_for_distance(".\\VacuoleAutophagySim\\testCenterFinding\\aioutput050.piff")
    aiDistanceTensor2, vol, aiCenters, WalID,_ = parse_voxel_file_for_distance(".\\VacuoleAutophagySim\\testCenterFinding\\aioutput050V2.piff")
    

    #t0 = perf_counter()

    #OneChannelDistances2,MultiChannelDistances2 = oneHotToDistance_chamfer(vol, aiCenters,10)
    #print(f"{perf_counter() - t0:.6f} s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vol = vol.to(device)

    t0 = perf_counter()
    OneChannelDistances3,MultiChannelDistances3 = oneHotToDistance_fast(vol, aiCenters)
    print(f"{perf_counter() - t0:.6f} s")

    #t0 = perf_counter()
    #OneChannelDistances,MultiChannelDistances = oneHotToDistance(vol, aiCenters)
    #print(f"{perf_counter() - t0:.6f} s")

    smoothedDistance = smoothDistance(MultiChannelDistances3)

    tmp1,tmp2 = masked_gradient3d(smoothedDistance, inward_bias=-1.25)

    smoothedDistance = condense_single_channel(tmp1)

    smoothedDistance = smoothedDistance * -1

    #t,_ = point_direct_to_centers(smoothedDistance, step_mode="sign")


    tmp = smoothedDistance.clone()
    #quiver_slice_zyx(tmp,  axis='y', index=98, stride=1, savePath="C:\\Users\\evans\\tmp\\Addedv0.png" )
    
    mask = None
    while mask == None or len(mask[(mask<100) & (mask>0)]) > 0:
        print("--------------------------------------------------------------------------------------------")
        a,b,c = tmp[0,199,98,100].item(), tmp[1,199,98,100].item(), tmp[2,199,98,100].item()
        rz, ry, rx = round(199 + a), round(98 + b), round(100 + c)
        a2,b2,c2 = smoothedDistance[0,rz, ry, rx].item(), smoothedDistance[1,rz, ry, rx].item(), smoothedDistance[2,rz, ry, rx].item()
        print(a, b, c)
        print(a2,b2,c2)
        print("--------------------------------------------------------------------------------------------")
        tmp, mask = sum_with_next_from(tmp, tmp, neighbor_vals=tmp, avoid_self=True, mask=mask)
        print(mask[(mask<100) & (mask>0)].shape)
        #quiver_slice_zyx(tmp,  axis='z', index=98, stride=1, savePath="C:\\Users\\evans\\tmp\\Addedv"+str(i+1)+".png" )

    tmp = snap_vectors_to_nearest_non_nan(tmp,search_radius=9)
    #tmp, _ = remove_vectors_pointing_to_nan(tmp,tmp,torch.nan)

    mask = None
    while mask == None or len(mask[(mask<100) & (mask>0)]) > 0:
        print("--------------------------------------------------------------------------------------------")
        a,b,c = tmp[0,199,98,100].item(), tmp[1,199,98,100].item(), tmp[2,199,98,100].item()
        rz, ry, rx = round(199 + a), round(98 + b), round(100 + c)
        a2,b2,c2 = smoothedDistance[0,rz, ry, rx].item(), smoothedDistance[1,rz, ry, rx].item(), smoothedDistance[2,rz, ry, rx].item()
        print(a, b, c)
        print(a2,b2,c2)
        print("--------------------------------------------------------------------------------------------")
        tmp, mask = sum_with_next_from(tmp, tmp, neighbor_vals=tmp, avoid_self=True, mask=mask)
        print(mask[(mask<100) & (mask>0)].shape)


    tmp = snap_vectors_to_nearest_non_nan(tmp,search_radius=9)
    
    mask = None
    while mask == None or len(mask[(mask<100) & (mask>0)]) > 0:
        print("--------------------------------------------------------------------------------------------")
        a,b,c = tmp[0,199,98,100].item(), tmp[1,199,98,100].item(), tmp[2,199,98,100].item()
        rz, ry, rx = round(199 + a), round(98 + b), round(100 + c)
        a2,b2,c2 = smoothedDistance[0,rz, ry, rx].item(), smoothedDistance[1,rz, ry, rx].item(), smoothedDistance[2,rz, ry, rx].item()
        print(a, b, c)
        print(a2,b2,c2)
        print("--------------------------------------------------------------------------------------------")
        tmp, mask = sum_with_next_from(tmp, tmp, neighbor_vals=smoothedDistance, avoid_self=True, mask=mask)
        print(mask[(mask<100) & (mask>0)].shape)


    #tmp, _ = remove_vectors_pointing_to_nan(tmp,tmp,torch.nan)

    tmp = snap_vectors_to_nearest_non_nan(tmp)


    coords_idx = displacements_to_coords(tmp, round_to_int=True)
    triplets = coords_idx.permute(1, 2, 3, 0)
    triplets = triplets[~torch.isnan(triplets[:,:,:,0]) & ~torch.isnan(triplets[:,:,:,1]) & ~torch.isnan(triplets[:,:,:,2])]
    triplets = triplets.reshape(-1,3)

    unique_triplets, counts = torch.unique(triplets, dim=0, return_counts=True) # how many map to each voxel
    
    result = torch.cat([counts.unsqueeze(1), unique_triplets], dim=1)
    
    centers = drop_nearby_by_count(result, radius=2.0, minCount=0)
    idx = torch.argsort(centers[:, 0], descending=True)
    centers_sorted = centers[idx]
    centers_sorted = centers_sorted[:32, :]

    matches, ai_unmatched, gt_unmatched = coord_match_and_unmatched(aiCenters, f)
    print(matches)
    print(ai_unmatched)
    print(gt_unmatched.to(torch.int32))

    counts.max()  # largest cluster size


    quiver_slice_zyx(DistanceTensor.squeeze(0),  axis='y', index=98, stride=1)
    quiver_slice_zyx(DistanceTensor2.squeeze(0),  axis='y', index=98, stride=1)
    quiver_slice_zyx(aiDistanceTensor.squeeze(0),  axis='y', index=98, stride=1)
    quiver_slice_zyx(aiDistanceTensor2.squeeze(0),  axis='y', index=98, stride=1)
    print("End")

    






def main():
    ap = argparse.ArgumentParser(
        description="Load PIFF with your loader, find EDT max per ID, save YZ slice PNG at each center X."
    )
    ap.add_argument("piffs", type=Path, nargs="+", help="Path(s) to .piff file(s)")
    ap.add_argument("--outdir", type=Path, default=None, help="Output directory (default: alongside each .piff)")
    ap.add_argument("--show", action="store_true", help="Show slices interactively after saving")
    #args = ap.parse_args()
    piffs = [".\\VacuoleAutophagySim\\testCenterFinding\\output000.piff", ".\\VacuoleAutophagySim\\testCenterFinding\\output050.piff", ".\\VacuoleAutophagySim\\testCenterFinding\\output100.piff", ".\\VacuoleAutophagySim\\testCenterFinding\\output150.piff", ".\\VacuoleAutophagySim\\testCenterFinding\\output200.piff", ".\\VacuoleAutophagySim\\testCenterFinding\\output250.piff", ".\\VacuoleAutophagySim\\testCenterFinding\\output300.piff"]

    process_file(piffs, None, False)


if __name__ == "__main__":
    main()

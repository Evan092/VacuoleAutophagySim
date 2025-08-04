import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import torch
from Constants import noise_dim
from Utils import parse_voxel_file
# or use your preferred divergence function

def load_distribution(path: Path):
    # return a 1D array of voxel counts or probabilities
    one_hot_volume = parse_voxel_file(path)

        # sanity check
    assert one_hot_volume.ndim == 4 and one_hot_volume.shape[0] == 3, \
        "Expected shape (3, D, H, W)"
    
    # argmax over the channel dimension to get a (D, H, W) label map
    label_map = np.argmax(one_hot_volume, axis=0)
    
    # flatten to a 1D vector
    return label_map.ravel()

from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from tqdm import tqdm
import torch

@lru_cache(maxsize=None)
def load_cc3d(path_str):
    # cache each CC3D distribution as a NumPy array
    return load_distribution(Path(path_str))

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from tqdm import tqdm
import torch

def runAccuracyTest(model, device):
    model.eval()
    results = []
    root = Path(r"D:\runs")

    for run_dir in tqdm(sorted(root.iterdir()), desc="Runs"):
        # 1) Preload all CC3D distributions per step
        cc3d_cache = { step: [] for step in range(100) }
        for i in range(1, 6):
            out_dir = run_dir / f"outputs_{i:02d}"
            for step in range(100):
                p = out_dir / f"output{step:03d}.piff"
                cc3d_cache[step].append(load_cc3d(str(p)))

        # 2) Preload all input volumes per step (as CPU tensors)
        inp_cache = { step: [] for step in range(100) }
        for i in range(1, 6):
            out_dir = run_dir / f"outputs_{i:02d}"
            for step in range(100):
                arr = parse_voxel_file(out_dir / f"output{step:03d}.piff").unsqueeze(0)
                # shape: (1, C, D, H, W)
                inp_cache[step].append(arr)

        # 3) For each step, batch the 5 inputs into one forward pass
        for step in tqdm(range(100), desc=f"{run_dir.name} steps", leave=False):
            # stack to (5, C, D, H, W) and move to device
            inp_batch = torch.cat(inp_cache[step], dim=0).to(device)
            # single noise batch (5, noise_dim)
            z = torch.randn(inp_batch.size(0), noise_dim, device=device) * 0.1

            with torch.no_grad():
                out_batch = model(inp_batch, z, steps=step+1)  # (5, C, D, H, W)

            # move all 5 outputs to CPU+NumPy at once
            gan_np = out_batch.cpu().numpy().reshape(out_batch.size(0), -1)

            # compare each of the 5 GAN runs vs each of the 5 CC3D runs
            metric_vals = []
            for gan_vec in gan_np:
                for cc3d_vec in cc3d_cache[step]:
                    metric_vals.append(ks_2samp(gan_vec, cc3d_vec).statistic)

            results.append({
                "run":  run_dir.name,
                "step": step,
                "score": float(np.mean(metric_vals)),
            })

    df = pd.DataFrame(results)
    summary = df.groupby("step")["score"].agg(["mean","std","count"])
    print(summary)

def count_files(dir_path):
    # Only top-level files:
    return sum(
        1 for name in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, name))
    )

def runControlAccuracyTest(number):
    results = []
    root = Path(r"D:\AccuracyRuns\stepCount" + str(number))

    cc3d_cache = {}
    cc3d_cache2 = {}

    filesPerSet = count_files(root)
    for i in range(0, filesPerSet/2):
        p = root / f"output{i:03d}.piff"
        cc3d_cache.append(load_cc3d(str(p)))

    for i in range(filesPerSet/2, filesPerSet):
        p = root / f"output{i:03d}.piff"
        cc3d_cache2.append(load_cc3d(str(p)))

    metric_vals = []
    metric_vals.append(ks_2samp(cc3d_cache, cc3d_cache).statistic)

    #results.append({
    #    "run":  run_dir.name,
    #    "step": step,
    #    "score": float(np.mean(metric_vals)),
    #})

    #df = pd.DataFrame(results)
    #summary = df.groupby("step")["score"].agg(["mean","std","count"])
    #print(summary)
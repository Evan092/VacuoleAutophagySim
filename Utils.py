
from datetime import datetime
import numpy as np


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

def pad_crop_resize(vol, target_size=150):
    """
    vol: np.ndarray, shape (3, D, H, W)
    returns: vol padded/cropped/resized so spatial dims == target_size
    """
    C, D, H, W = vol.shape
    t = target_size

    # 1) PAD any dim < t, equally on both sides
    pad_z = max(t - D, 0)
    pad_y = max(t - H, 0)
    pad_x = max(t - W, 0)
    # split pads
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

    # 2) CROP any dim > t, but only from zero‐regions so we never lose a 1
    def crop_axis(arr, axis):
        # arr: boolean mask where True indicates occupied (Body or Wall)
        nonzero = np.any(arr, axis=tuple(i for i in range(arr.ndim) if i != axis))
        first, last = np.where(nonzero)[0][[0, -1]]
        L = arr.shape[axis]
        excess = L - t
        if excess <= 0:
            return slice(None)  # no crop
        # how many zeros before/after the body region?
        zeros_before = first
        zeros_after = L - 1 - last
        if zeros_before + zeros_after >= excess:
            # remove as evenly as possible from before/after zeros
            remove_before = min(zeros_before, excess // 2)
            remove_after = excess - remove_before
            start = remove_before
            end = L - remove_after
            return slice(start, end)
        else:
            # not enough zeros to crop—skip cropping this axis
            return slice(None)

    # build occupancy mask from channels 1&2
    occ = (vol[1] + vol[2]) > 0

    # determine slices for D, H, W
    sz = crop_axis(occ, axis=0)
    sy = crop_axis(occ, axis=1)
    sx = crop_axis(occ, axis=2)

    vol = vol[:, sz, sy, sx]

    # 3) if still not exactly t×t×t, resize with nearest‐neighbor
    _, D2, H2, W2 = vol.shape
    if not (D2 == H2 == W2 == t):
        factors = (1,
                   t / D2,
                   t / H2,
                   t / W2)
        vol = zoom(vol, zoom=factors, order=0)  # order=0 → nearest‐neighbor

    return vol


def parse_voxel_file(path):
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
            _, cell_type, x1, x2, y1, y2, z1, z2 = parts
            x1, x2 = int(x1), int(x2)
            y1, y2 = int(y1), int(y2)
            z1, z2 = int(z1), int(z2)
            voxels.append((cell_type, x1, x2, y1, y2, z1, z2))
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)
            max_z = max(max_z, z2)

    # 2) Build initial volume
    vol = np.zeros((3, max_z + 1, max_y + 1, max_x + 1), dtype=np.float32)
    for cell_type, x1, x2, y1, y2, z1, z2 in voxels:
        if cell_type == "Body":
            vol[1, z1 : z2 + 1, y1 : y2 + 1, x1 : x2 + 1] = 1.0
        elif cell_type == "Wall":
            vol[2, z1 : z2 + 1, y1 : y2 + 1, x1 : x2 + 1] = 1.0

    vol = pad_crop_resize(vol, target_size=132)

    # 4) Fill the “empty” channel
    mask_occupied = (vol[1] + vol[2]) == 0
    vol[0, mask_occupied] = 1.0

    return vol






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

    return vol

def parse_voxel_file_for_ID_matching(path):
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



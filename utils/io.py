import numpy as np

def load_bin(file_path):
    points = np.fromfile(file_path, dtype=np.float32)
    points = points.reshape(-1, 4)

    xyz = points[:, :3]
    intensity = points[:, 3]

    return xyz, intensity
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from plyfile import PlyData
from torch.utils.data import Dataset


class PointCloudDataset(Dataset):
    """Loads .ply point clouds with paired .labels.npy label files."""

    def __init__(self, root: Path, max_points: int = 4096) -> None:
        self.root = Path(root)
        self.max_points = max_points
        self.cloud_files: List[Path] = sorted(self.root.glob("*.ply"))
        if not self.cloud_files:
            raise FileNotFoundError(f"No .ply files found in {self.root}")

    def __len__(self) -> int:
        return len(self.cloud_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cloud_path = self.cloud_files[idx]
        label_path = cloud_path.with_suffix(".labels.npy")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file for {cloud_path.name}: {label_path}")

        xyz = load_xyz_from_ply(cloud_path)
        labels = np.load(label_path).astype(np.int64)

        if xyz.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Point/label mismatch for {cloud_path.name}: {xyz.shape[0]} vs {labels.shape[0]}"
            )

        xyz, labels = random_sample_points(xyz, labels, self.max_points)
        return torch.from_numpy(xyz).float(), torch.from_numpy(labels).long()


def load_xyz_from_ply(path: Path) -> np.ndarray:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    return xyz.astype(np.float32)


def random_sample_points(
    xyz: np.ndarray, labels: np.ndarray, max_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    n = xyz.shape[0]
    if n <= max_points:
        return xyz, labels

    idx = np.random.choice(n, size=max_points, replace=False)
    return xyz[idx], labels[idx]

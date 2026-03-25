# filename: csv_loader.py
# author: AK Wash
# date: 2026-03-24
# description: loads just the xyz values from the in situ LiDAR scan
# for processing by inference

from pathlib import Path
import numpy as np


def load_csv_point_cloud(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float32, encoding=None)

    if data.dtype.names is None:
        raise ValueError("CSV file has no header row.")

    required_cols = ("X1_mm", "Y1_mm", "Z1_mm")
    if not all(col in data.dtype.names for col in required_cols):
        raise ValueError(
            f"CSV file must contain columns {required_cols}, found {data.dtype.names}"
        )

    points = np.column_stack((
        data["X1_mm"],
        data["Y1_mm"],
        data["Z1_mm"]
    )).astype(np.float32)

    # remove rows where xyz are all zero
    mask = ~np.all(points == 0, axis=1)

    points = points / 1000.0
    points = points[mask]

    # workspace: 15 ft × 15 ft
    workspace_size_m = 15 * 0.3048   # 4.572 m
    half_size = workspace_size_m / 2 # 2.286 m

    # height filter (remove ceiling / far wall noise)
    min_z = -2.0
    max_z = 6

    mask_crop = (
        (points[:, 0] >= -half_size) & (points[:, 0] <= half_size) &
        (points[:, 1] >= -half_size) & (points[:, 1] <= half_size) &
        (points[:, 2] >= min_z) & (points[:, 2] <= max_z)
    )

    points = points[mask_crop]
    print(f"Points after crop: {points.shape[0]}")

    # no extra features; use empty array for now
    features = np.zeros((points.shape[0], 0), dtype=np.float32)

    return points, features
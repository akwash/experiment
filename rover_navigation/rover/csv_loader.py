# filename: csv_loader.py
# author: AK Wash
# date: 2026-03-24
# description: loads just the xyz values from the in situ LiDAR scan
# for processing by inference

from pathlib import Path
import numpy as np
from rover_navigation.perception.dataset import _normalize_points


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


    # convert ot meters from mm
    points = points / 1000 
    
    # # remove rows of NaN
    # NaN_mask = np.isfinite(points).all(axis=1)
    # NaN_mask &= ~np.all(np.isclose(points,0.0),axis=1)
    # points = points[NaN_mask]

    # # print("Any NaNs left:", np.isnan(points).any())

    # # remove rows where xyz are all zero
    # mask = ~np.all(points == 0, axis=1)

    # points = points
    # points = points[mask]


    # no extra features; use empty array for now
    features = np.zeros((points.shape[0], 0), dtype=np.float32)

    return points, features
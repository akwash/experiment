# filename: csv_loader.py
# author: AK Wash
# date: 2026-03-24
# description: loads just the xyz values from the in situ LiDAR scan
# for processing by inference

from pathlib import Path
import numpy as np

def load_csv_point_cloud(
        path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float32)

    required_cols = ("x","y","z")
    if data.dytpe.names is None:
        raise ValueError("CSV file has no header row.")
    
    if not all(col in data.dytpe.names for col in required_cols):
        raise ValueError(
            f"CSV file must contain columns {required_cols}, found {data.dype.names}"
        )
    
    points = np.column_stack((data["x"], data["y"], data["z"])).astype(np.float32)
    features = np.zeros((points.shape[0],0), dtype=np.float32)

    return points, features
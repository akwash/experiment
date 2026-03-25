# Filename: load_ply.py
# Author: AK Wash
# Created: 2026-03-10

# Description: loads the CloudCompare binary ply files. Reads the
# file and extracts:
# - 3D point coords
# - annotation labels

# returns three arrays:
# - points   -> shape (N,3)
# - features -> shape (N,0)
# - labels   -> shape (N,)

# used in: preprocessing

from pathlib import Path
from typing import Any

from plyfile import PlyData
import numpy as np
import yaml


# load function for yaml files, used to read the metadata fields from the ply files
# input: path or string
# output: dictionary with string keys and values
def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)  # if given string convert to path object

    # open file at path in read mode, automatically close after
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)  # convert yaml to python objects


def load_cloudcompare_ply(
    path: str | Path,
    dataset_config_path: str | Path = "rover_navigation/config/dataset.yaml",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    load cloud compare ply file
    :param path: path to ply file
    :param dataset_config_path: unused here, kept for compatibility
    :return: points, features, labels
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"PLY file not found: {path}")

    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    field_names = vertex.dtype.names

    if field_names is None:
        raise ValueError(f"No vertex fields found in {path}")

    # coordinates
    required_xyz = ("x", "y", "z")
    if not all(name in field_names for name in required_xyz):
        raise ValueError(
            f"PLY file must contain fields {required_xyz}, found {field_names}"
        )

    points = np.vstack([
        vertex["x"],
        vertex["y"],
        vertex["z"],
    ]).T.astype(np.float32)

    # the file is set in mm for xyz
    points /= 1000.0

    # labels
    if "scalar_label" not in field_names:
        raise ValueError(
            f"PLY file must contain 'scalar_label'. Found fields: {field_names}"
        )

    labels = np.asarray(vertex["scalar_label"]).astype(np.int64)

    # no extra features used
    features = np.zeros((points.shape[0], 0), dtype=np.float32)

    # basic shape checks
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points shape (N, 3), got {points.shape}")

    if features.ndim != 2:
        raise ValueError(f"Expected features shape (N, F), got {features.shape}")

    if labels.ndim != 1:
        raise ValueError(f"Expected labels shape (N,), got {labels.shape}")

    if not (len(points) == len(features) == len(labels)):
        raise ValueError(
            f"Length mismatch: points={len(points)}, features={len(features)}, labels={len(labels)}"
        )

    return points, features, labels
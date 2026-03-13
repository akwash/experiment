# Filename: load_ply.py
# Author: AK Wash
# Created: 2026-03-10

# Description: loads the CloudCompare binary ply files. Reads the 
# file and extracts: 
# - 3D point coords
# - scalar field vals
#   - intensity
#   - segmentation ID
#   - annotation labels
#   - metadata fields

# returns two arrays:
# - points -> shape (N,3)
# - scalars -> shape (N,S)

# used in: preprocessing
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
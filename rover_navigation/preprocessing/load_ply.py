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
from plyfile import PlyData
import numpy as np
import yaml 

# load function for yaml files, used to read the metadata fields from the ply files
# input: path or string
# output: dictionary with string keys and values
def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path) # if given string convert to path object

    # open file at math in read mode, automatically close after
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) # convert yaml to python objects
    

def load_cloudcompare_ply(path: str | Path):
    """
    load cloud compare ply file
    :param path: path to ply file
    :return: points and scalars
    """

    path = Path(path)

    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    field_names = vertex.dtype.names

    # coordiantes
    points = np.vstack([
        vertex["x"], 
        vertex["y"], 
        vertex["z"]
        ]).T.astype(np.float32) # xyz vals
    
    # RGB features
    features = None 
    if all(name in field_names for name in ("red", "green", "blue")):
        features = np.vstack([
            vertex["red"],
            vertex["green"],
            vertex["blue"]
        ]).T.astype(np.float32)

    # labels
    labels = None 
    if "scalar_label" in field_names:
        labels = np.asarray(vertex["scalar_label"]).astype(np.int32)
                                                    
    return points, features, labels

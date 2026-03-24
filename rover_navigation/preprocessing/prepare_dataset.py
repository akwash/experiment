# Filename: prepare_dataset.py
# Author: AK Wash
# Created: 2026-03-10

# Description: prepares dataset, converts the raw '.ply' files
# into a "training" ready dataset for the model

# 1. Loads the point cloud
# 2. extracts the annotation label
# 3. saves processed data into the compressed .npz file

# output files contain:
# points -> (N,3) XYZ coordinates
# features -> (N,F) scalar feature values
# labels -> (N,) class labels

# used in: data/processed and dataset loader

from pathlib import Path

import numpy as np

from rover_navigation.preprocessing.load_ply import load_cloudcompare_ply
from rover_navigation.util.config_loader import load_yaml


# input: raw .ply, output path
# output: saves the processed dataset as a compressed .npz file
def prepare_one_file(
    input_path: str | Path,
    output_path: str | Path,
    dataset_config_path: str | Path = "rover_navigation/config/dataset.yaml",
) -> None:
    # load in the numpy arrays from the .ply file
    points, features, labels = load_cloudcompare_ply(
        input_path,
        dataset_config_path=dataset_config_path,
    )

    # check shapes
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

    # make sure labels are binary obstacle / non-obstacle
    unique_labels = np.unique(labels)
    if not np.all(np.isin(unique_labels, [0, 1])):
        raise ValueError(
            f"Expected binary labels in {{0,1}}, got {unique_labels.tolist()}"
        )

    # create output directory
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # make the compressed .npz file with points, features, and labels
    np.savez_compressed(
        output_path,
        points=points.astype(np.float32),
        features=features.astype(np.float32),
        labels=labels.astype(np.int64),
    )


# load in the configuration yaml for dataset
def main() -> None:
    dataset_cfg = load_yaml("rover_navigation/config/dataset.yaml")

    paths_cfg = dataset_cfg["paths"]  # input/output path

    raw_dir = Path(paths_cfg["raw_dir"])               # folder containing raw .ply files
    processed_dir = Path(paths_cfg["processed_dir"])   # save processed files here

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    processed_dir.mkdir(parents=True, exist_ok=True)

    # find all .ply files in raw directory
    ply_files = sorted(raw_dir.glob("*.ply"))

    if not ply_files:
        raise ValueError(f"No .ply files found in {raw_dir}")

    print(f"Found {len(ply_files)} .ply file(s) in {raw_dir}")

    # process each file
    for input_path in ply_files:
        file_stem = input_path.stem
        output_path = processed_dir / f"{file_stem}.npz"

        prepare_one_file(
            input_path,
            output_path,
            dataset_config_path="rover_navigation/config/dataset.yaml",
        )
        print(f"Saved processed dataset to {output_path}")


if __name__ == "__main__":
    main()
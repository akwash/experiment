# Filename: prepare_dataset.py
# Author: AK Wash
# Created: 2026-03-10

# Description: prepares dataset, converts the raw '.ply' files
# into a "training" ready dataset for the model

# 1. Loads the point cloud 
# 2. seperates the scalar fields into input features and albels
# 3. extracts the annotation label
# 4. save processed data into the compressed .npz file

# output files contain:
# points -> (N,3) XYZ coordinates
# features -> (N,F) scalar feature values
# labels -> (N,) class labels

# used in: data/processed and dataset loader
import os
from pathlib import Path

import numpy as np

from preprocessing.load_ply import load_cloudcompare_ply
from util.config_loader import load_yaml

# input: raw .ply, output path, label_column
# output: saves the processed dataset as a compressed .npz file
def prepare_one_file(
    input_path: str,
    output_path: str,
    label_column: int,
) -> None:
    # load in the two numpy arrays from the .ply file
    points, scalars = load_cloudcompare_ply(input_path)

    # cheeck that it is a 2D matrix
    if scalars.ndim != 2:
        raise ValueError(f"Expected scalars to be 2D, got shape {scalars.shape}")

    # check that the label column index is valid
    if label_column < 0 or label_column >= scalars.shape[1]:
        raise ValueError(
            f"Label column {label_column} is out of bounds for scalars with shape {scalars.shape}"
        )

    labels = scalars[:, label_column].astype(np.int64) # get label column and convert to integer class label
    feature_cols = [i for i in range(scalars.shape[1]) if i != label_column] # seperate the feature columns from the label column
    features = scalars[:, feature_cols].astype(np.float32) # convert features to float32

    # create output directory
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # make the compressed .npz file with points, features, and labels
    np.savez_compressed(
        output_path,
        points=points.astype(np.float32),
        features=features,
        labels=labels,
    )

# load in the configuration yaml for dataset
def main() -> None:
    dataset_cfg = load_yaml("config/dataset.yaml")

    paths_cfg = dataset_cfg["paths"] # input/output path
    pc_cfg = dataset_cfg["point_cloud"] # label index

    input_path = paths_cfg["test_file"] # which file to process
    processed_dir = paths_cfg["processed_dir"] # save processed file here
    label_column = pc_cfg["label_scalar_index"] # which scalar field is the label

    # file naming
    file_stem = Path(input_path).stem
    output_path = os.path.join(processed_dir, f"{file_stem}.npz")

    # run processing, confirm saved file
    prepare_one_file(input_path, output_path, label_column)
    print(f"Saved processed dataset to {output_path}")


if __name__ == "__main__":
    main()
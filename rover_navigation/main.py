# Filename: main.py
# Author: AK Wash
# Created: 2026-03-10

# Description: Performs the following:
# 1. Load LiDAR point cloud scan.
# 2. Run semantic segmentation using RandLA-Net.
# 3. Produce predicted obstacle labels for each point.
# 4. Produces traversability map
# 5. Terrain cost scoring
# 6. Path planning
# 7. Rover navigation

import numpy as np

from perception.infer import run_inference
from util.config_loader import load_yaml


def main() -> None:
    dataset_cfg = load_yaml("config/dataset.yaml")
    ply_path = dataset_cfg["paths"]["test_file"]

    points, true_labels, pred_labels = run_inference(ply_path)

    print("Pipeline ran successfully.")
    print(f"Points shape: {points.shape}")
    print(f"True labels unique: {np.unique(true_labels)}")
    print(f"Pred labels unique: {np.unique(pred_labels)}")


if __name__ == "__main__":
    main()
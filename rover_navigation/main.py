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

from pathlib import Path
import numpy as np

from rover_navigation.util.config_loader import load_yaml
from rover_navigation.planning.run_navigation import run_navigation

# create the file paths so that it doesnt matter where its run from
ROOT = Path(__file__).resolve().parent
DATASET_CONFIG = ROOT / "config" / "dataset.yaml"

def main() -> None
    # load the dataset configuration
    dataset_cfg = load_yaml(DATASET_CONFIG)
    ply_path = dataset_cfg["paths"]["test_file"]

    # rover start and goal in the world coordiantes (meters)
    # NEED TO UPDATE FOR ACTUAL MAP
    rover_pose_xy = (0.0,0.0)
    goal_pose_xy = (5.0,5.0)

    # navigation pipeline
    path = run_navigation(
        ply_path=ply_path,
        rover_pose_xy=rover_pose_xy
        goal_pose_xy=goal_pose_xy,
    )

    print("\nNativgation Complete.")
    print(f"Path length: {len(path)}")
    print("PathL")
    for step in path:
        print(step)

if __name__ == "__main__":
    main()
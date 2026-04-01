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
import matplotlib.pyplot as plt

from rover_navigation.util.config_loader import load_yaml
from rover_navigation.perception.infer import run_inference
from rover_navigation.mapping.occupancy_map import (
    SLAM,
    build_map_from_predictions,
    grid_to_world,
    transform_to_world,
    world_to_grid,
)
from rover_navigation.planning.dstar_lite import DStarLite
from rover_navigation.rover.position_finder import heading_from_path


# create the file paths so that it doesnt matter where its run from
ROOT = Path(__file__).resolve().parent
DATASET_CONFIG = ROOT / "config" / "dataset.yaml"


def main() -> None:
    # load the dataset configuration
    dataset_cfg = load_yaml(DATASET_CONFIG)
    ply_path = dataset_cfg["paths"]["test_file"]

    print(f"\nUsing point cloud: {ply_path}")

    # rover start and goal in the world coordiantes (meters)
    # NEED TO UPDATE FOR ACTUAL MAP
    # allowable range x: 0 to 4.4 m, 0 to <15 ft
    # allowable range y: 0 to 4.4 m, 0 to < 15 ft
    rover_pose_xy = (3 * 0.3048, 3 * 0.3048)
    goal_pose_xy = (9 * 0.3048, 4 * 0.3048)

    
    # Run semantic segmentation using RandLA-Net
   
    print("\nRunning RandLA-Net inference...")
    points, true_labels, pred_labels = run_inference(ply_path)

    print("Inference complete.")
    print(f"Total points: {len(points)}")
    print(f"Unique predicted labels: {np.unique(pred_labels)}")

    # Build occupancy map from predictions
    
    print("\nBuilding occupancy map...")
    truth_map, grid_info = build_map_from_predictions(
        points,
        pred_labels,
        grid_resolution=0.1524,
        obstacle_label=1,
    )
    # Check to see if there are obstacle labels being passed through
    unique = np.unique(pred_labels)
    print("Unique predicted labels:", unique)

    contains_0 = 0 in unique
    contains_1 = 1 in unique

    print("Contains 0:", contains_0)
    print("Contains 1:", contains_1)
    num_ones = np.sum(pred_labels == 1)
    print("Number of ones:", num_ones)

    # inflate obstacles for rover safety margin
    truth_map.inflate(radius=2)

    grid = truth_map.get_map()


    # check to see if the obstacle cells are within the bounds of the map
    rows, cols = grid.shape
    obs_r, obs_c = np.where(grid == 1)
    invalid = np.where(
    (obs_r < 0) | (obs_r >= rows) |
    (obs_c < 0) | (obs_c >= cols)
    )[0]

    if len(invalid) > 0:
        print("ERROR: Some obstacle cells are out of bounds!")
        print("Example:", list(zip(obs_r[invalid][:10], obs_c[invalid][:10])))
    else:
        print("All obstacle cells fall within map bounds.")


    # debugging checks
    # Convert start and goal into grid cells
    start = world_to_grid(rover_pose_xy[0], rover_pose_xy[1], grid_info)
    goal = world_to_grid(goal_pose_xy[0], goal_pose_xy[1], grid_info)

    rows, cols = truth_map.get_map().shape

    print(f"Start world: {rover_pose_xy}")
    print(f"Goal world: {goal_pose_xy}")
    print(f"Start (grid): {start}")
    print(f"Goal (grid): {goal}")
    print("Grid shape:", (rows, cols))
    print("Grid info:", grid_info)

    max_x = grid_info["min_x"] + (cols - 1) * grid_info["resolution"]
    max_y = grid_info["min_y"] + (rows - 1) * grid_info["resolution"]

    print("Map x range:", grid_info["min_x"], "to", max_x)
    print("Map y range:", grid_info["min_y"], "to", max_y)

    for name, node in [("start", start), ("goal", goal)]:
        r, c = node
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(
                f"{name} {node} is outside map bounds. "
                f"Valid rows: 0..{rows-1}, cols: 0..{cols-1}"
            )
    
    # Run D* Lite planner
    print("\nRunning D* Lite planning...")
    planner = DStarLite(map=truth_map, s_start=start, s_goal=goal) # planner takes in inflated obstacles
    path, g, rhs = planner.move_and_replan(robot_position=start) 

    print("Planning complete.")
    print(f"Path length: {len(path)}")

    slam = SLAM(map=truth_map, view_range=2) 
    
    for step_idx, grid_pos in enumerate(path):
        heading = heading_from_path(path, step_idx) # get heading in world frame for the current step
        

    world_x, world_y = grid_to_world(grid_pos[0], grid_pos[1], grid_info) # convert grid to world coords
    points_world = transform_to_world(points, (world_x, world_y), heading) # transform point cloud to world frame based on current position

    # update the map with the new transformed scan
    new_map, _ = build_map_from_predictions(
        points_world,
        pred_labels,
        grid_resolution=0.1524,
        obstacle_label=1,
    )
    slam.slam_map.set_map(new_map.get_map()) # update the slam map with map from current scan

    changed_vertices, updated_map = slam.rescan(global_pos=grid_pos) # get the changed vertices

    if len(changed_vertices.vertices) > 0:
        print(f"New obstacles at step {step_idx}, replanning...")
        path, g, rhs = planner.move_and_replan(robot_position=grid_pos)

    grid = truth_map.get_map() # get final grid for visualization

    # Visualize occupancy map and path
    print("\nVisualizing results...")
    path_np = np.array(path)

    plt.figure(figsize=(8, 6))
    plt.imshow(grid, cmap="gray_r")

    # plot path
    plt.plot(path_np[:, 0], path_np[:, 1], "r-", linewidth=2, label="Path")

    # plot start and goal
    plt.scatter(path_np[0, 0], path_np[0, 1], c="green", s=80, label="Start")
    plt.scatter(path_np[-1, 0], path_np[-1, 1], c="blue", s=80, label="Goal")

    plt.title("Occupancy Grid + Planned Path")
    plt.xlabel("X (grid)")
    plt.ylabel("Y (grid)")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    plt.show()

    
    # Print path
    print("\nNavigation Complete.")
    print(f"Path length: {len(path)}")
    print("Path:")
    for step in path:
        print(step)


if __name__ == "__main__":
    main()
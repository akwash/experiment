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
    build_map_from_predictions,
    sensor_to_rover_local,
    transform_local_to_world,
    world_to_grid,
    OccupancyMap,
)
from rover_navigation.planning.dstar_lite import DStarLite
from rover_navigation.debug.debug_transport import DebugSender, UdpJsonSender
from rover_navigation.debug.rover_debug_sender import send_planning_debug


# create the file paths so that it doesnt matter where its run from
ROOT = Path(__file__).resolve().parent
DATASET_CONFIG = ROOT / "config" / "dataset.yaml"


def load_scan_sequence(scan_input: str | Path) -> list[Path]:
    """
    Load a sequence of CSV scans from a directory or a single CSV file path.
    Returns paths in sorted order for repeatable replay.
    """
    scan_input = Path(scan_input)
    if scan_input.is_file():
        if scan_input.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV scan file, got: {scan_input}")
        return [scan_input]

    if not scan_input.exists():
        raise FileNotFoundError(f"Scan path not found: {scan_input}")
    if not scan_input.is_dir():
        raise ValueError(f"Expected file or directory for scans, got: {scan_input}")

    scans = sorted(scan_input.glob("*.csv"))
    if not scans:
        raise ValueError(f"No CSV scans found in directory: {scan_input}")
    return scans


def fuse_scan_into_global_map(
    global_map: OccupancyMap | None,
    scan_map: OccupancyMap,
) -> OccupancyMap:
    """
    Fuse current scan map into persistent map via occupied-cell union.
    """
    if global_map is None:
        fused = OccupancyMap(
            x_dim=scan_map.x_dim,
            y_dim=scan_map.y_dim,
            movement_setting=scan_map.movement_setting,
        )
        fused.set_map(scan_map.get_map().copy())
        return fused

    if (global_map.x_dim != scan_map.x_dim) or (global_map.y_dim != scan_map.y_dim):
        raise ValueError("Map dimensions differ; cannot fuse scan map into global map.")

    fused_grid = np.maximum(global_map.get_map(), scan_map.get_map())
    global_map.set_map(fused_grid)
    return global_map


def move_rover_along_path(
    path: list[tuple[int, int]],
    current_pos: tuple[int, int],
    max_steps: int,
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """
    Simulate rover motion by advancing a limited number of path steps.
    Returns new rover grid position and the moved segment (including start).
    """
    if not path:
        return current_pos, [current_pos]
    if len(path) == 1:
        return path[0], [path[0]]

    steps = max(1, int(max_steps))
    move_count = min(steps, len(path) - 1)
    moved_segment = path[: move_count + 1]
    new_pos = moved_segment[-1]
    return new_pos, moved_segment


def build_debug_sender(dataset_cfg: dict) -> DebugSender | None:
    """
    Create optional debug sender from config.

    Expected optional config block:
      debug:
        enabled: false
        host: "127.0.0.1"
        port: 9876
    """
    debug_cfg = dataset_cfg.get("debug", {})
    if not bool(debug_cfg.get("enabled", False)):
        return None

    host = str(debug_cfg.get("host", "127.0.0.1"))
    port = int(debug_cfg.get("port", 9876))
    return UdpJsonSender(host=host, port=port)


def main() -> None:
    # load the dataset configuration
    dataset_cfg = load_yaml(DATASET_CONFIG)
    debug_sender = build_debug_sender(dataset_cfg)
    scan_input = dataset_cfg["paths"]["test_file"]
    scan_paths = load_scan_sequence(scan_input)

    print(f"\nUsing scan source: {scan_input}")
    print(f"Found {len(scan_paths)} scan file(s).")

    # rover start and goal in the world coordiantes (meters)
    # NEED TO UPDATE FOR ACTUAL MAP
    # allowable range x: 0 to 4.4 m, 0 to <15 ft
    # allowable range y: 0 to 4.4 m, 0 to < 15 ft
    rover_pose_xy = (0 * 0.3048, 4 * 0.3048)
    goal_pose_xy = (8 * 0.3048, 4 * 0.3048)

    # rover heading should come from localization/odometry
    rover_heading = 0.0
    steps_per_cycle = 3
    persistent_map: OccupancyMap | None = None
    current_grid_pos: tuple[int, int] | None = None
    goal_grid_pos: tuple[int, int] | None = None
    traveled_path: list[tuple[int, int]] = []
    final_path: list[tuple[int, int]] = []

    grid_info = None

    for scan_idx, scan_path in enumerate(scan_paths, start=1):
        print(f"\n[{scan_idx}/{len(scan_paths)}] Processing scan: {scan_path}")
        points_sensor, _true_labels, pred_labels = run_inference(scan_path)

        points_local = sensor_to_rover_local(points_sensor)
        points_world = transform_local_to_world(
            points_local=points_local,
            rover_pose_xy=rover_pose_xy,
            heading_rad=rover_heading,
        )

        scan_map, scan_grid_info = build_map_from_predictions(
            points_world,
            pred_labels,
            grid_resolution=0.1524,
            obstacle_label=1,
        )

        if grid_info is None:
            grid_info = scan_grid_info
            current_grid_pos = world_to_grid(rover_pose_xy[0], rover_pose_xy[1], grid_info)
            goal_grid_pos = world_to_grid(goal_pose_xy[0], goal_pose_xy[1], grid_info)
            print(f"Start (grid): {current_grid_pos}")
            print(f"Goal  (grid): {goal_grid_pos}")
        else:
            if scan_grid_info != grid_info:
                raise ValueError("Grid info changed between scans; expected fixed grid geometry.")

        persistent_map = fuse_scan_into_global_map(persistent_map, scan_map)

        planning_map = OccupancyMap(
            x_dim=persistent_map.x_dim,
            y_dim=persistent_map.y_dim,
            movement_setting=persistent_map.movement_setting,
        )
        planning_map.set_map(persistent_map.get_map().copy())
        planning_map.inflate(radius=2)

        assert current_grid_pos is not None
        assert goal_grid_pos is not None

        planner = DStarLite(map=planning_map, s_start=current_grid_pos, s_goal=goal_grid_pos)
        path, _g, _rhs = planner.move_and_replan(robot_position=current_grid_pos)
        final_path = path

        if debug_sender is not None:
            send_planning_debug(
                sender=debug_sender,
                step_idx=scan_idx,
                heading_rad=rover_heading,
                occupancy_grid=planning_map.get_map(),
                path=path,
                rover_cell=current_grid_pos,
                goal_cell=goal_grid_pos,
            )

        current_grid_pos, moved_segment = move_rover_along_path(
            path=path,
            current_pos=current_grid_pos,
            max_steps=steps_per_cycle,
        )
        if not traveled_path:
            traveled_path.extend(moved_segment)
        else:
            traveled_path.extend(moved_segment[1:])

        print(f"Planned path length: {len(path)}")
        print(f"Moved to: {current_grid_pos}")

        if current_grid_pos == goal_grid_pos:
            print("Goal reached.")
            break

    if persistent_map is None:
        raise RuntimeError("No scans were processed.")

    grid = persistent_map.get_map()

    # Visualize occupancy map and path history
    print("\nVisualizing results...")
    path_np = np.array(final_path) if final_path else np.empty((0, 2), dtype=int)
    traveled_np = np.array(traveled_path) if traveled_path else np.empty((0, 2), dtype=int)

    plt.figure(figsize=(8, 6))
    plt.imshow(grid, cmap="gray_r")

    # plot latest planned path (x=col, y=row)
    if len(path_np) > 0:
        plt.plot(path_np[:, 1], path_np[:, 0], "r--", linewidth=2, label="Planned Path")

    # plot traveled trajectory (x=col, y=row)
    if len(traveled_np) > 0:
        plt.plot(traveled_np[:, 1], traveled_np[:, 0], "g-", linewidth=2, label="Traveled")
        plt.scatter(traveled_np[0, 1], traveled_np[0, 0], c="green", s=80, label="Start")
        plt.scatter(traveled_np[-1, 1], traveled_np[-1, 0], c="blue", s=80, label="Current")

    plt.title("Occupancy Grid + Scan/Plan Loop")
    plt.xlabel("X (col)")
    plt.ylabel("Y (row)")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nNavigation Complete.")
    print(f"Final path length: {len(final_path)}")
    print("Final path:")
    for step in final_path:
        print(step)


if __name__ == "__main__":
    main()
# Filename: main.py
# Author: AK Wash
# Created: 2026-03-10

# Description: Performs the following:
# 1. Load LiDAR points (CSV, PLY, or numpy .npy/.npz — see dataset.yaml paths.scan_mode).
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
from rover_navigation.perception.infer import run_inference, run_inference_from_xyz
from rover_navigation.mapping.occupancy_map import (
    build_map_from_predictions,
    grid_to_world,
    sensor_to_rover_local,
    transform_local_to_world,
    world_to_grid,
    OccupancyMap,
)
from rover_navigation.planning.dstar_lite import DStarLite
from rover_navigation.debug.debug_transport import DebugSender, UdpJsonSender
from rover_navigation.debug.rover_debug_sender import send_planning_debug
from rover_navigation.rover.motion import update_rover_pose_from_motion


# create the file paths so that it doesnt matter where its run from
ROOT = Path(__file__).resolve().parent
DATASET_CONFIG = ROOT / "config" / "dataset.yaml"


def load_file_scan_sequence(scan_input: str | Path, scan_glob: str = "*.csv") -> list[Path]:
    """
    Load a sequence of point-cloud files: single .csv / .ply or a directory of matches.

    :param scan_glob: when ``scan_input`` is a directory, glob pattern (e.g. ``*.csv``, ``*.ply``).
    :return: list of path objects to process in order
    """
    scan_input = Path(scan_input) # path object for handling
    if scan_input.is_file():
        suf = scan_input.suffix.lower() # suffix for single file
        if suf not in (".csv", ".ply"):
            raise ValueError(f"Expected a .csv or .ply scan file, got: {scan_input}")
        return [scan_input]

    if not scan_input.exists(): 
        raise FileNotFoundError(f"Scan path not found: {scan_input}")
    if not scan_input.is_dir():
        raise ValueError(f"Expected file or directory for scans, got: {scan_input}")

    scans = sorted(scan_input.glob(scan_glob)) # glob for multiple files
    if not scans:
        raise ValueError(
            f"No files matching {scan_glob!r} in directory: {scan_input}"
        )
    return scans


def load_xyz_numpy_file(path: str | Path) -> np.ndarray:
    """
    Load (N, 3) xyz from ``.npy`` or ``.npz`` (keys ``xyz`` or ``points``).
    :param path: file path
    :return: (N,3) float array for xyz points 
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Numpy scan not found: {path}")

    suf = path.suffix.lower() # suffix for load method
    if suf == ".npy":
        arr = np.load(path, allow_pickle=False) # load .npy
    elif suf == ".npz":
        z = np.load(path) # load .npz
        names = z.files
        if "xyz" in names:
            arr = z["xyz"] 
        elif "points" in names:
            arr = z["points"]
        else:
            raise ValueError(
                f"NPZ {path} must contain array 'xyz' or 'points'; got {list(names)}"
            )
    else:
        raise ValueError(f"Expected .npy or .npz, got: {path}")

    arr = np.asarray(arr, dtype=np.float32) # ensure float32
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected xyz shape (N, 3), got {arr.shape}") # check shape
    return arr


def load_numpy_scan_sequence(paths_cfg: dict) -> list[tuple[np.ndarray, str]]:
    """
    Build a list of (xyz_array, label) from ``paths`` numpy settings.
    :param paths_cfg: config dict for paths (dataset.yaml)
    """
    single = paths_cfg.get("numpy_scan") # signle file path
    scan_dir = paths_cfg.get("numpy_scan_dir") # directory for multiple files
    glob_pat = str(paths_cfg.get("numpy_scan_glob", "*.npy")) # glob pattern for directory

    out: list[tuple[np.ndarray, str]] = [] # list of (xyz, label)

    if single:
        p = Path(single)
        out.append((load_xyz_numpy_file(p), str(p)))
    elif scan_dir:
        d = Path(scan_dir)
        if not d.is_dir():
            raise NotADirectoryError(f"numpy_scan_dir is not a directory: {d}")
        for p in sorted(d.glob(glob_pat)):
            if p.is_file():
                out.append((load_xyz_numpy_file(p), str(p)))
    else:
        raise ValueError(
            "scan_mode numpy requires paths.numpy_scan or paths.numpy_scan_dir"
        )

    if not out:
        raise ValueError("No numpy scans found (check glob and directory).")
    return out


def resolve_scans(dataset_cfg: dict) -> list[tuple[str, Path | np.ndarray, str]]:
    """
    Decide how to load scans from ``dataset_cfg['paths']``.

    Returns a list of (kind, payload, label) where:
      - kind ``file``: payload is a Path to csv/ply for :func:`run_inference`
      - kind ``numpy``: payload is (N,3) float32 for :func:`run_inference_from_xyz`
    """
    paths_cfg = dataset_cfg["paths"] # config block for path
    mode = str(paths_cfg.get("scan_mode", "file")).lower() # mode for loading scans

    if mode == "numpy":
        return [
            ("numpy", xyz, label) for xyz, label in load_numpy_scan_sequence(paths_cfg)
        ]

    if mode == "file":
        scan_glob = str(paths_cfg.get("scan_glob", "*.csv"))
        test_file = paths_cfg.get("test_file")
        if not test_file:
            raise ValueError("paths.test_file is required when scan_mode is file")
        file_paths = load_file_scan_sequence(test_file, scan_glob=scan_glob)
        return [("file", p, str(p)) for p in file_paths]

    raise ValueError(f"Unknown paths.scan_mode: {mode!r} (use 'file' or 'numpy')")


def fuse_scan_into_global_map(
    global_map: OccupancyMap | None,
    scan_map: OccupancyMap,
) -> OccupancyMap:
    """
    Fuse current scan map into persistent map via occupied-cell union.
    :param global_map: existing map to update, None starts new
    :param scan_map: new scan to fuse (need same dimensions as global_map)
    """
    if global_map is None:
        fused = OccupancyMap(
            x_dim=scan_map.x_dim,
            y_dim=scan_map.y_dim,
            movement_setting=scan_map.movement_setting,
        )
        fused.set_map(scan_map.get_map().copy()) # start with scan if no global map
        return fused

    if (global_map.x_dim != scan_map.x_dim) or (global_map.y_dim != scan_map.y_dim):
        raise ValueError("Map dimensions differ; cannot fuse scan map into global map.")

    fused_grid = np.maximum(global_map.get_map(), scan_map.get_map()) # union of occupied cells
    global_map.set_map(fused_grid) # update global map with the fused result
    return global_map


def move_rover_along_path(
    path: list[tuple[int, int]],
    current_pos: tuple[int, int],
    max_steps: int,
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """
    Simulate rover motion by advancing a limited number of path steps.
    Returns new rover grid position and the moved segment (including start).
    :param path: full planned path (row,col) grid cells
    :param current_pos: current rover pos (row,col)
    :param max_steps: max number of steps to move along the path
    :return: (new_pos, moved_segment) new pos is updated position, moved_segment list of cells traversed
    """
    if not path:
        return current_pos, [current_pos]
    if len(path) == 1:
        return path[0], [path[0]]

    steps = max(1, int(max_steps)) # at least one step if path
    move_count = min(steps, len(path) - 1) # don't go past path length
    moved_segment = path[: move_count + 1] # include current pos + steps
    new_pos = moved_segment[-1] # last cell in moved segement is the new position
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
    debug_cfg = dataset_cfg.get("debug", {}) # config block for the debug settings
    if not bool(debug_cfg.get("enabled", False)):
        return None

    host = str(debug_cfg.get("host", "127.0.0.1")) # default to local host
    port = int(debug_cfg.get("port", 9876)) # default port for sender
    return UdpJsonSender(host=host, port=port) 


def main() -> None:
    # load the dataset configuration
    dataset_cfg = load_yaml(DATASET_CONFIG) 
    debug_sender = build_debug_sender(dataset_cfg)
    scans = resolve_scans(dataset_cfg) # load scans based on settings

    print(f"\nUsing scan_mode: {dataset_cfg['paths'].get('scan_mode', 'file')}")
    print(f"Found {len(scans)} scan(s).")

    # rover start and goal in the world coordiantes (meters)
    # NEED TO UPDATE FOR ACTUAL MAP
    # allowable range x: 0 to 4.4 m, 0 to <15 ft
    # allowable range y: 0 to 4.4 m, 0 to < 15 ft
    rover_pose_xy = (0 * 0.3048, 4 * 0.3048)
    goal_pose_xy = (8 * 0.3048, 4 * 0.3048)

    # inital rover state set after 1st scan
    rover_heading = 0.0
    steps_per_cycle = 3 # take 3 path steps before rescan
    persistent_map: OccupancyMap | None = None # global map for scan info
    current_grid_pos: tuple[int, int] | None = None # rover pos in (row, col)
    goal_grid_pos: tuple[int, int] | None = None # goal pos in (row, col)
    traveled_path: list[tuple[int, int]] = [] # history of rover pos (row, col)
    final_path: list[tuple[int, int]] = [] # last planned path (row, col)

    grid_info = None # store grid info from first scan

    for scan_idx, (scan_kind, scan_payload, scan_label) in enumerate(scans, start=1):
        print(f"\n[{scan_idx}/{len(scans)}] Processing scan: {scan_label}")
        if scan_kind == "file":
            points_sensor, _true_labels, pred_labels = run_inference(scan_payload)
        elif scan_kind == "numpy":
            points_sensor, _true_labels, pred_labels = run_inference_from_xyz(
                scan_payload
            )
        else:
            raise RuntimeError(f"Unknown scan kind: {scan_kind}")

        points_local = sensor_to_rover_local(points_sensor) # convert sensor cords to local frame
        points_world = transform_local_to_world(
            points_local=points_local,
            rover_pose_xy=rover_pose_xy,
            heading_rad=rover_heading,
        )

        # build occupancy map from predictions and world points
        scan_map, scan_grid_info = build_map_from_predictions(
            points_world,
            pred_labels,
            grid_resolution=0.1524,
            obstacle_label=1,
        )

        # intialize grid info and positions
        if grid_info is None:
            grid_info = scan_grid_info
            current_grid_pos = world_to_grid(rover_pose_xy[0], rover_pose_xy[1], grid_info) # convert rover world pos to grid pos
            goal_grid_pos = world_to_grid(goal_pose_xy[0], goal_pose_xy[1], grid_info) # convert goal world pos to grid pos
            print(f"Start (grid): {current_grid_pos}")
            print(f"Goal  (grid): {goal_grid_pos}")
        else:
            if scan_grid_info != grid_info:
                raise ValueError("Grid info changed between scans; expected fixed grid geometry.")

        persistent_map = fuse_scan_into_global_map(persistent_map, scan_map) # update global map with new scan

        planning_map = OccupancyMap(
            x_dim=persistent_map.x_dim,
            y_dim=persistent_map.y_dim,
            movement_setting=persistent_map.movement_setting,
        )
        planning_map.set_map(persistent_map.get_map().copy()) # use fused map for planning
        planning_map.inflate(radius=2) # inflate obstacles (Safety margin)

        assert current_grid_pos is not None
        assert goal_grid_pos is not None

        planner = DStarLite(map=planning_map, s_start=current_grid_pos, s_goal=goal_grid_pos) # intialize path planner
        path, _g, _rhs = planner.move_and_replan(robot_position=current_grid_pos) # plan path
        final_path = path # store final path

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

        # keep rover pose and yaw consistent with simulated motion
        rover_pose_xy, rover_heading = update_rover_pose_from_motion(
            rover_pose_xy=rover_pose_xy,
            rover_heading=rover_heading,
            moved_segment=moved_segment,
            grid_info=grid_info,
        )

        print(f"Planned path length: {len(path)}")
        print(f"Moved to: {current_grid_pos}")
        print(f"Rover world pose: {rover_pose_xy}, heading(rad): {rover_heading:.3f}")

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
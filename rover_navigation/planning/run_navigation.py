from rover_navigation.perception.infer import run_inference
from rover_navigation.mapping.occupancy_map import (
    build_map_from_predictions,
    sensor_to_rover_local,
    transform_local_to_world,
    world_to_grid,
    SLAM,
)
from rover_navigation.planning.dstar_lite import DStarLite


def run_navigation(
    ply_path: str,
    rover_pose_xy: tuple[float, float],
    goal_pose_xy: tuple[float, float],
    rover_heading: float = 0.0,
):
    # perception
    points, true_labels, pred_labels = run_inference(ply_path)
    points_local = sensor_to_rover_local(points)
    points_world = transform_local_to_world(points_local, rover_pose_xy, rover_heading)

    # get the map from predictions
    truth_map, grid_info = build_map_from_predictions(
        points_world,
        pred_labels,
        grid_resolution=0.10,
        obstacle_label=1,
    )

    # inflate obstacles for rover clearance
    truth_map.inflate(radius=2)

    # put the start and end goal on grid
    start = world_to_grid(rover_pose_xy[0], rover_pose_xy[1], grid_info)
    goal = world_to_grid(goal_pose_xy[0], goal_pose_xy[1], grid_info)

    # initialize the planner
    planner = DStarLite(map=truth_map, s_start=start, s_goal=goal)
    path, g, rhs = planner.move_and_replan(robot_position=start)

    # initialize SLAM manager
    slam = SLAM(map=truth_map, view_range=3)

    current = start

    while current != goal:
        changed_vertices, slam_map = slam.rescan(global_pos=current)

        if changed_vertices.vertices:
            planner.new_edges_and_old_costs = changed_vertices
            path, g, rhs = planner.move_and_replan(robot_position=current)

        if len(path) < 2:
            print("No path found.")
            break

        current = path[1]
        print("Move to:", current)

    return path
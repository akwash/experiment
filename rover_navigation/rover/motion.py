from __future__ import annotations

import numpy as np

from rover_navigation.mapping.occupancy_map import grid_to_world


def update_rover_pose_from_motion(
    rover_pose_xy: tuple[float, float],
    rover_heading: float,
    moved_segment: list[tuple[int, int]],
    grid_info: dict,
) -> tuple[tuple[float, float], float]:
    """
    Update rover world pose and heading from simulated grid motion.

    Cell tuples are treated as (row, col) in this loop.
    - Position uses final cell via grid_to_world(row, col).
    - Heading is computed in world coordinates from the final moved edge.
    """
    if not moved_segment:
        return rover_pose_xy, rover_heading

    end_row, end_col = moved_segment[-1]
    rover_pose_xy = grid_to_world(end_row, end_col, grid_info)

    if len(moved_segment) >= 2:
        prev_row, prev_col = moved_segment[-2]
        prev_x, prev_y = grid_to_world(prev_row, prev_col, grid_info)
        end_x, end_y = rover_pose_xy
        dx = end_x - prev_x
        dy = end_y - prev_y
        if dx != 0 or dy != 0:
            rover_heading = float(np.arctan2(dy, dx))

    return rover_pose_xy, rover_heading

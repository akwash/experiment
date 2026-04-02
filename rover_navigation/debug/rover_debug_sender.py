from __future__ import annotations

import numpy as np

from rover_navigation.debug.debug_protocol import DebugFrame
from rover_navigation.debug.debug_transport import DebugSender


def send_planning_debug(
    sender: DebugSender,
    step_idx: int,
    heading_rad: float,
    occupancy_grid: np.ndarray,
    path: list[tuple[int, int]],
    rover_cell: tuple[int, int],
    goal_cell: tuple[int, int],
) -> None:
    """
    Build and send one lightweight planning debug frame.

    Payload fields:
      - occupancy grid
      - current path
      - rover and goal cells
      - rover heading
      - step index
    """
    frame = DebugFrame(
        step_idx=step_idx,
        heading_rad=heading_rad,
        occupancy_grid=occupancy_grid,
        path=path,
        rover_cell=rover_cell,
        goal_cell=goal_cell,
    )
    sender.send(frame)

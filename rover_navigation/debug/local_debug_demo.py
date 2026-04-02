from __future__ import annotations

import threading
import time

import numpy as np

from rover_navigation.debug.debug_transport import create_in_memory_transport_pair
from rover_navigation.debug.live_viewer import LiveDebugViewer
from rover_navigation.debug.rover_debug_sender import send_planning_debug


def _run_fake_rover_sender() -> None:
    sender, receiver = create_in_memory_transport_pair()
    viewer = LiveDebugViewer(receiver)

    def producer() -> None:
        grid = np.zeros((40, 40), dtype=np.uint8)
        grid[20, 8:32] = 1
        goal = (35, 35)
        heading = 0.0
        for step in range(1, 80):
            rover = (min(step, 35), min(step, 35))
            path = [(r, r) for r in range(rover[0], goal[0] + 1)]
            send_planning_debug(
                sender=sender,
                step_idx=step,
                heading_rad=heading,
                occupancy_grid=grid,
                path=path,
                rover_cell=rover,
                goal_cell=goal,
            )
            time.sleep(0.08)

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    viewer.run(poll_interval_s=0.03)


if __name__ == "__main__":
    _run_fake_rover_sender()

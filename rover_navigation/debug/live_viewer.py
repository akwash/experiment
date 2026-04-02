from __future__ import annotations

import argparse
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from rover_navigation.debug.debug_protocol import DebugFrame
from rover_navigation.debug.debug_transport import (
    DebugReceiver,
    UdpJsonReceiver,
)


class LiveDebugViewer:
    """Matplotlib viewer for live planning debug frames."""

    def __init__(self, receiver: DebugReceiver):
        self.receiver = receiver # set receiver to get debug frames from
        self._latest: Optional[DebugFrame] = None # store latest frame

        self.fig, self.ax = plt.subplots(figsize=(8, 6)) # create figure adn axis
        self.im = None # image handle for occupancy grid
        self.path_line = None # line handle for path
        self.rover_scatter = None # scatter handle for rover pos
        self.goal_scatter = None # scatter handle for goal pos
        self.ax.set_title("Live Rover Planning Debug")
        self.ax.set_xlabel("X (grid col)")
        self.ax.set_ylabel("Y (grid row)")

    def _draw_frame(self, frame: DebugFrame) -> None:
        grid = np.asarray(frame.occupancy_grid)

        if self.im is None:
            self.im = self.ax.imshow(
                grid,
                cmap="gray_r",
                origin="lower",  # world convention: +x right, +y forward
                interpolation="nearest",
            )
        else:
            self.im.set_data(grid)

        if self.path_line is not None:
            self.path_line.remove()
            self.path_line = None
        if frame.path:
            path_np = np.asarray(frame.path, dtype=int)
            self.path_line = self.ax.plot(
                path_np[:, 1],  # x = col
                path_np[:, 0],  # y = row
                "r-",
                linewidth=2,
                label="Path",
            )[0]

        if self.rover_scatter is not None:
            self.rover_scatter.remove()
        if self.goal_scatter is not None:
            self.goal_scatter.remove()

        self.rover_scatter = self.ax.scatter(
            frame.rover_cell[1], frame.rover_cell[0], c="lime", s=80, label="Rover"
        )
        self.goal_scatter = self.ax.scatter(
            frame.goal_cell[1], frame.goal_cell[0], c="cyan", s=80, label="Goal"
        )

        self.ax.legend(loc="upper right")
        self.ax.set_title(
            f"Live Rover Planning Debug | step={frame.step_idx} | heading={frame.heading_rad:.3f} rad"
        )
        self.fig.canvas.draw_idle()

    def run(self, poll_interval_s: float = 0.05) -> None:
        """
        Start live viewer update loop until window closes.
        """
        plt.ion()
        plt.show(block=False)
        while plt.fignum_exists(self.fig.number):
            newest = self.receiver.recv_latest()
            if newest is not None:
                self._latest = newest
                self._draw_frame(newest)
            plt.pause(poll_interval_s)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Live planning debug viewer.")
    p.add_argument("--bind-host", default="0.0.0.0", help="UDP bind host")
    p.add_argument("--bind-port", type=int, default=9876, help="UDP bind port")
    p.add_argument("--poll-s", type=float, default=0.05, help="Viewer poll interval seconds")
    return p


def main() -> int:
    args = _build_arg_parser().parse_args()
    receiver = UdpJsonReceiver(args.bind_host, args.bind_port)
    viewer = LiveDebugViewer(receiver)
    viewer.run(poll_interval_s=args.poll_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

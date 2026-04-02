from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np


@dataclass
class DebugFrame:
    """
    Lightweight planning debug payload.

    Coordinates are grid cells in (row, col) order internally.
    Viewers should plot as x=col, y=row.
    """

    step_idx: int
    heading_rad: float
    occupancy_grid: np.ndarray
    path: list[tuple[int, int]]
    rover_cell: tuple[int, int]
    goal_cell: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        grid = np.asarray(self.occupancy_grid) # numpy array for validation 
        if grid.ndim != 2:
            raise ValueError(f"Expected 2D occupancy grid, got shape {grid.shape}")

        return {
            "step_idx": int(self.step_idx),
            "heading_rad": float(self.heading_rad),
            "occupancy_grid": grid.astype(np.uint8).tolist(),
            "path": [[int(r), int(c)] for r, c in self.path],
            "rover_cell": [int(self.rover_cell[0]), int(self.rover_cell[1])],
            "goal_cell": [int(self.goal_cell[0]), int(self.goal_cell[1])],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DebugFrame":
        return cls(
            step_idx=int(payload["step_idx"]),
            heading_rad=float(payload["heading_rad"]),
            occupancy_grid=np.asarray(payload["occupancy_grid"], dtype=np.uint8),
            path=[(int(rc[0]), int(rc[1])) for rc in payload["path"]],
            rover_cell=(int(payload["rover_cell"][0]), int(payload["rover_cell"][1])),
            goal_cell=(int(payload["goal_cell"][0]), int(payload["goal_cell"][1])),
        )

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "DebugFrame":
        return cls.from_dict(json.loads(data.decode("utf-8")))

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


@dataclass(frozen=True)
class GoalCommand:
    """
    Operator command to update the planner goal.

    Notes:
      - Internal planner coordinates are (row, col).
      - For this first version, the operator enters goal as (row, col).
    """

    goal_cell: tuple[int, int]

    command_type: str = "set_goal"

    def to_dict(self) -> dict[str, Any]:
        row, col = self.goal_cell
        return {"type": self.command_type, "goal_cell": [row, col]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GoalCommand":
        if payload.get("type") != "set_goal":
            raise ValueError(f"Unsupported command type: {payload.get('type')!r}")

        goal_cell = payload.get("goal_cell")
        if (
            not isinstance(goal_cell, (list, tuple))
            or len(goal_cell) != 2
        ):
            raise ValueError("goal_cell must be [row, col]")

        row, col = goal_cell[0], goal_cell[1]
        if not (isinstance(row, int) and isinstance(col, int)):
            raise ValueError("goal_cell values must be integers")

        return cls(goal_cell=(row, col))

    def to_json_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, data: bytes) -> "GoalCommand":
        payload = json.loads(data.decode("utf-8"))
        return cls.from_dict(payload)


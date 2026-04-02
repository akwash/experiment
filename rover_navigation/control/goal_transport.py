from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import socket

from rover_navigation.control.goal_protocol import GoalCommand


class GoalSender(ABC):
    """Abstract goal sender interface."""

    @abstractmethod
    def send_goal(self, goal_command: GoalCommand) -> None:
        """Send one goal command."""


class GoalReceiver(ABC):
    """Abstract goal receiver interface."""

    @abstractmethod
    def recv_latest(self) -> Optional[GoalCommand]:
        """
        Return the latest received goal command, or None if nothing arrived.

        Must not block the planning loop.
        """


class UdpJsonGoalSender(GoalSender):
    """UDP JSON goal sender (laptop -> Jetson)."""

    def __init__(self, host: str, port: int):
        self._target = (host, int(port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_goal(self, goal_command: GoalCommand) -> None:
        self._sock.sendto(goal_command.to_json_bytes(), self._target)


class UdpJsonGoalReceiver(GoalReceiver):
    """UDP JSON goal receiver (Jetson side)."""

    def __init__(self, bind_host: str, bind_port: int, timeout_s: float = 0.001):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((bind_host, int(bind_port)))
        self._sock.settimeout(float(timeout_s))

    def recv_latest(self) -> Optional[GoalCommand]:
        latest: Optional[GoalCommand] = None
        while True:
            try:
                data, _addr = self._sock.recvfrom(64 * 1024)
                latest = GoalCommand.from_json_bytes(data)
            except socket.timeout:
                break
            except Exception:
                # Ignore malformed packets so the planner loop stays robust.
                break
        return latest


from __future__ import annotations

from abc import ABC, abstractmethod
from queue import Empty, Queue
import socket
from typing import Optional

from rover_navigation.debug.debug_protocol import DebugFrame


class DebugSender(ABC):
    """Transport-agnostic sender interface."""

    @abstractmethod
    def send(self, frame: DebugFrame) -> None:
        """Send one debug frame."""


class DebugReceiver(ABC):
    """Transport-agnostic receiver interface."""

    @abstractmethod
    def recv_latest(self) -> Optional[DebugFrame]:
        """
        Return the latest available frame, or None when no frame is available.
        Implementations may drop stale intermediate frames.
        """


class InMemorySender(DebugSender):
    """Local loopback sender for single-process testing."""

    def __init__(self, queue: Queue[DebugFrame]):
        self._queue = queue

    def send(self, frame: DebugFrame) -> None:
        self._queue.put(frame)


class InMemoryReceiver(DebugReceiver):
    """Local loopback receiver for single-process testing."""

    def __init__(self, queue: Queue[DebugFrame]):
        self._queue = queue

    def recv_latest(self) -> Optional[DebugFrame]:
        latest: Optional[DebugFrame] = None
        while True:
            try:
                latest = self._queue.get_nowait()
            except Empty:
                break
        return latest


class UdpJsonSender(DebugSender):
    """Simple UDP JSON sender for rover -> laptop debug streaming."""

    def __init__(self, host: str, port: int):
        self._target = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, frame: DebugFrame) -> None:
        self._sock.sendto(frame.to_json_bytes(), self._target)


class UdpJsonReceiver(DebugReceiver):
    """Simple UDP JSON receiver for live debug visualization."""

    def __init__(self, bind_host: str, bind_port: int, timeout_s: float = 0.001):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP socket
        self._sock.bind((bind_host, bind_port)) # bind to specified host/port
        self._sock.settimeout(timeout_s) # set timeout for non-blocking recv

    def recv_latest(self) -> Optional[DebugFrame]:
        latest: Optional[DebugFrame] = None # latest frame received
        while True:
            try:
                data, _addr = self._sock.recvfrom(10 * 1024 * 1024) # receive data
                latest = DebugFrame.from_json_bytes(data) # parse JSON
            except socket.timeout:
                break
        return latest


def create_in_memory_transport_pair() -> tuple[InMemorySender, InMemoryReceiver]:
    """Factory for local sender/receiver testing without networking."""
    q: Queue[DebugFrame] = Queue()
    return InMemorySender(q), InMemoryReceiver(q)

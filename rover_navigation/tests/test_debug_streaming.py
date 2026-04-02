import numpy as np

from rover_navigation.debug.debug_protocol import DebugFrame
from rover_navigation.debug.debug_transport import create_in_memory_transport_pair


def test_debug_frame_json_roundtrip():
    frame = DebugFrame(
        step_idx=5,
        heading_rad=1.25,
        occupancy_grid=np.array([[0, 1], [1, 0]], dtype=np.uint8),
        path=[(0, 0), (1, 1)],
        rover_cell=(0, 0),
        goal_cell=(1, 1),
    )

    restored = DebugFrame.from_json_bytes(frame.to_json_bytes())
    assert restored.step_idx == frame.step_idx
    assert np.isclose(restored.heading_rad, frame.heading_rad)
    assert np.array_equal(restored.occupancy_grid, frame.occupancy_grid)
    assert restored.path == frame.path
    assert restored.rover_cell == frame.rover_cell
    assert restored.goal_cell == frame.goal_cell


def test_in_memory_receiver_returns_latest_frame():
    sender, receiver = create_in_memory_transport_pair()

    sender.send(
        DebugFrame(
            step_idx=1,
            heading_rad=0.0,
            occupancy_grid=np.zeros((3, 3), dtype=np.uint8),
            path=[(0, 0)],
            rover_cell=(0, 0),
            goal_cell=(2, 2),
        )
    )
    sender.send(
        DebugFrame(
            step_idx=2,
            heading_rad=0.5,
            occupancy_grid=np.ones((3, 3), dtype=np.uint8),
            path=[(1, 1), (2, 2)],
            rover_cell=(1, 1),
            goal_cell=(2, 2),
        )
    )

    latest = receiver.recv_latest()
    assert latest is not None
    assert latest.step_idx == 2
    assert np.isclose(latest.heading_rad, 0.5)
    assert latest.rover_cell == (1, 1)

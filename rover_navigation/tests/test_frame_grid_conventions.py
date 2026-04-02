import numpy as np

from rover_navigation.mapping.occupancy_map import (
    grid_to_world,
    sensor_to_rover_local,
    transform_local_to_world,
    world_to_grid,
)


def test_sensor_to_rover_local_ouster_axis_mapping():
    # Ouster: +X forward, +Y left -> local: +X right, +Y forward
    # Expected mapping: x_local=-y_sensor, y_local=x_sensor
    sensor_points = np.array(
        [
            [1.0, 0.0, 0.0],   # front
            [0.0, 1.0, 0.0],   # left
            [0.0, 0.0, 1.0],   # up
        ],
        dtype=np.float32,
    )
    local_points = sensor_to_rover_local(sensor_points)
    expected = np.array(
        [
            [0.0, 1.0, 0.0],   # front -> +Y
            [-1.0, 0.0, 0.0],  # left -> -X (right-positive frame)
            [0.0, 0.0, 1.0],   # up unchanged
        ],
        dtype=np.float32,
    )
    assert np.allclose(local_points, expected, atol=1e-6)


def test_transform_local_to_world_heading_and_translation():
    local_points = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)  # +X in local
    rover_pose_xy = (2.0, 3.0)

    # heading 0: no rotation
    world_h0 = transform_local_to_world(local_points, rover_pose_xy, 0.0)
    assert np.allclose(world_h0, np.array([[3.0, 3.0, 0.0]], dtype=np.float32), atol=1e-6)

    # heading +90deg: +X local rotates to +Y world
    world_h90 = transform_local_to_world(local_points, rover_pose_xy, np.pi / 2.0)
    assert np.allclose(world_h90, np.array([[2.0, 4.0, 0.0]], dtype=np.float32), atol=1e-6)


def test_world_grid_roundtrip_cell_origin_consistency():
    grid_info = {"resolution": 0.5, "min_x": 1.0, "min_y": 2.0}
    row, col = world_to_grid(2.0, 3.0, grid_info)
    assert (row, col) == (2, 2)

    x, y = grid_to_world(row, col, grid_info)
    assert np.isclose(x, 2.0)
    assert np.isclose(y, 3.0)

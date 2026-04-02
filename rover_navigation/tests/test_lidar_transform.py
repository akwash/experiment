import numpy as np

from rover_navigation.mapping.occupancy_map import transform_to_world


def test_lidar_to_world_known_point():
    """
    A point directly in front of rover maps to +Y in world.

    Setup:
      - Ouster LiDAR front point: [1, 0, 0]
      - rover world pose: origin
      - heading 0 rad
      - default LiDAR->body mapping:
          x_body = -y_lidar
          y_body =  x_lidar

    Expected:
      lidar [1,0,0] -> world [0,1,0]
    """
    point_lidar = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    rover_pose_xy = (0.0, 0.0)
    heading_rad = 0.0

    point_world = transform_to_world(point_lidar, rover_pose_xy, heading_rad)
    expected = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

    assert np.allclose(point_world, expected, atol=1e-6)


def test_transform_uses_configurable_lidar_extrinsics():
    """
    Non-default lidar_to_body_rotation and translation are respected.
    """
    points_lidar = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    rover_pose_xy = (0.0, 0.0)
    heading_rad = 0.0

    # Custom mapping: x_body=x_lidar, y_body=y_lidar, plus x translation +2.
    lidar_to_body_rotation = np.eye(3, dtype=np.float32)
    lidar_to_body_translation = np.array([2.0, 0.0, 0.0], dtype=np.float32)

    point_world = transform_to_world(
        points_lidar,
        rover_pose_xy,
        heading_rad,
        lidar_to_body_rotation=lidar_to_body_rotation,
        lidar_to_body_translation=lidar_to_body_translation,
    )
    expected = np.array([[3.0, 0.0, 0.0]], dtype=np.float32)
    assert np.allclose(point_world, expected, atol=1e-6)

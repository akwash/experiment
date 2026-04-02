import numpy as np

from rover_navigation.mapping.occupancy_map import transform_to_world


def test_lidar_to_world_known_point():
    """
    Verify one known LiDAR point transforms as expected.

    Setup:
      - LiDAR point 1m forward in LiDAR frame: [0, 1, 0]
      - rover world pose: (2, 3), yaw=+90 deg
      - default LiDAR->body mapping: [x_b, y_b] = [y_l, -x_l]

    Expected:
      lidar [0,1,0] -> body [1,0,0]
      yaw +90 deg: body [1,0,0] -> world delta [0,1,0]
      add rover pose -> world [2,4,0]
    """
    point_lidar = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    rover_pose_xy = (2.0, 3.0)
    heading_rad = np.pi / 2.0

    point_world = transform_to_world(point_lidar, rover_pose_xy, heading_rad)
    expected = np.array([[2.0, 4.0, 0.0]], dtype=np.float32)

    assert np.allclose(point_world, expected, atol=1e-6)

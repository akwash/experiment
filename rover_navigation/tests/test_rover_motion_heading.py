import numpy as np

from rover_navigation.rover.motion import update_rover_pose_from_motion


def test_update_rover_pose_heading_axis_aligned():
    grid_info = {"resolution": 1.0, "min_x": 0.0, "min_y": 0.0}
    rover_pose_xy = (0.0, 0.0)
    rover_heading = 0.0

    # move one cell in +x world direction: (row, col) (0,0) -> (0,1)
    moved_segment = [(0, 0), (0, 1)]
    pose, heading = update_rover_pose_from_motion(
        rover_pose_xy=rover_pose_xy,
        rover_heading=rover_heading,
        moved_segment=moved_segment,
        grid_info=grid_info,
    )
    assert np.allclose(pose, (1.0, 0.0), atol=1e-6)
    assert np.isclose(heading, 0.0, atol=1e-6)

    # move one cell in +y world direction: (row, col) (0,1) -> (1,1)
    moved_segment = [(0, 1), (1, 1)]
    pose, heading = update_rover_pose_from_motion(
        rover_pose_xy=pose,
        rover_heading=heading,
        moved_segment=moved_segment,
        grid_info=grid_info,
    )
    assert np.allclose(pose, (1.0, 1.0), atol=1e-6)
    assert np.isclose(heading, np.pi / 2.0, atol=1e-6)


def test_update_rover_pose_heading_diagonal():
    grid_info = {"resolution": 0.5, "min_x": 0.0, "min_y": 0.0}
    rover_pose_xy = (0.0, 0.0)
    rover_heading = 0.0

    # diagonal move: (row, col) (2,2) -> (3,3) means dx=+0.5, dy=+0.5
    moved_segment = [(2, 2), (3, 3)]
    pose, heading = update_rover_pose_from_motion(
        rover_pose_xy=rover_pose_xy,
        rover_heading=rover_heading,
        moved_segment=moved_segment,
        grid_info=grid_info,
    )
    assert np.allclose(pose, (1.5, 1.5), atol=1e-6)
    assert np.isclose(heading, np.pi / 4.0, atol=1e-6)

from rover_navigation.mapping.occupancy_map import OccupancyMap, SLAM, OBSTACLE, UNOCCUPIED

def test_rescan_updates_slam():
    gt = OccupancyMap(5,5)
    gt.set_obstacles((2,2))

    slam = SLAM(gt, view_range = 1)
    vertices, slam_map = slam.rescan((2,2))

    assert not slam_map.is_unoccupied((2,2))
    assert len(vertices.vertices) >= 1
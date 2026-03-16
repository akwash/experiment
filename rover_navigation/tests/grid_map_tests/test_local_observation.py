from mapping.occupancy_map import OccupancyMap, OBSTACLE, UNOCCUPIED

def test_local_observation_obstacles():
    m = OccupancyMap(5,5)
    m.set_obstacle((2,2))

    obs = m.local_observation((2,2), view_range = 1)
    assert obs[(2,2)] == OBSTACLE
    assert obs[(1,1)] == UNOCCUPIED

    
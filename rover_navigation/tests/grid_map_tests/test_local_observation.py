from rover_navigation.mapping.occupancy_map import OccupancyMap, OBSTACLE, UNOCCUPIED

def test_local_observation_obstacles():
    m = OccupancyMap(5,5)
    m.set_obstacles((2,2))

    obs = m.observations((2,2), view_range = 1)
    assert obs[(2,2)] == OBSTACLE
    assert obs[(1,1)] == UNOCCUPIED

    
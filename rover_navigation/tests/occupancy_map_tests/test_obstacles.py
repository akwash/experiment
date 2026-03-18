from rover_navigation.mapping.occupancy_map import OccupancyMap, OBSTACLE, UNOCCUPIED

def test_set_obstacle():
    m = OccupancyMap(5,5)
    m.set_obstacles((1,1))
    assert m.occupancy_map[1,1] == OBSTACLE

def test_remove_obstacle():
    m = OccupancyMap(5,5)
    m.set_obstacles((1,1))
    assert m.occupancy_map[1,1] == OBSTACLE

    m.remove_obstacle((1,1))
    assert m.occupancy_map[1,1] == UNOCCUPIED
    

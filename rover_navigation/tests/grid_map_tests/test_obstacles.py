from mapping.grid_map import OccupancyGridMap, OBSTACLE, UNOCCUPIED

def test_set_obstacle():
    m = OccupancyGridMap(5,5)
    m.set_obstacle((1,1))
    assert m.occpancy_grid_map[1,1] == OBSTACLE

def test_remove_obstacle():
    m = OccupancyGridMap(5,5)
    m.set_obstacle((1,1))
    assert m.occpancy_grid_map[1,1] == OBSTACLE

    m.remove_obstacle((1,1))
    assert m.occupancy_grid_map[1,1] == UNOCCUPIED
    

from mapping.grid_map import OccupancyGridMap, OBSTACLE, UNOCCUPIED

def test_in_bounds():
    m = OccupancyGridMap(5,5)
    assert m.in_bounds((0,0))
    assert m.in_bounds((4,4))
    assert not m.in_bounds((-1,0))
    assert not m.in_bounds((5,2))

def test_is_unoccupied():
    m = OccupancyGridMap(5,5)
    m.occupancy_grid_map[2,2] = OBSTACLE
    assert m.is_unoccupied((1,1))
    assert not m.is_unoccupied((2,2))
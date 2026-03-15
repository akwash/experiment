import numpy as np
from mapping.grid_map import OccupancyGridMap, OBSTACLE, UNOCCUPIED

def test_init_create_grid():
    m = OccupancyGridMap(5,4)
    assert m.xdim == 5
    assert m.ydim == 4
    assert m.map_extents == (5,4)
    assert m.occupancy_grid_map.shape == (5,4)
    assert np.all(m.occupancy_grid_map == UNOCCUPIED)

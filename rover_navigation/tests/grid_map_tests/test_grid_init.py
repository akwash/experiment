import numpy as np
from mapping.occupancy_map import OccupancyMap, OBSTACLE, UNOCCUPIED

def test_init_create_grid():
    m = OccupancyMap(5,4)
    assert m.xdim == 5
    assert m.ydim == 4
    assert m.map_boundaries == (5,4)
    assert m.occupancy_occupancy_map.shape == (5,4)
    assert np.all(m.occupancy_occupancy_map == UNOCCUPIED)

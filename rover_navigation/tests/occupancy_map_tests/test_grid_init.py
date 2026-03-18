import numpy as np
from rover_navigation.mapping.occupancy_map import OccupancyMap, OBSTACLE, UNOCCUPIED

def test_init_create_grid():
    m = OccupancyMap(5,4)
    assert m.x_dim == 5
    assert m.y_dim == 4
    assert m.map_boundaries == (5,4)
    assert m.occupancy_map.shape == (5,4)
    assert np.all(m.occupancy_map == UNOCCUPIED)

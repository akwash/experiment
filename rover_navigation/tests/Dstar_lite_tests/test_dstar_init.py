import numpy as np
from rover_navigation.planning.dstar_lite import DStarLite
from rover_navigation.mapping.occupancy_map import OccupancyMap, OBSTACLE, UNOCCUPIED

def test_dstar_init():
    m = OccupancyMap(5,5)
    dstar = DStarLite(
        map=m,
        s_start=(0,0),
        s_goal=(4,4),
    )
    
    assert dstar.s_start == (0,0)
    assert dstar.s_goal == (4,4)
    assert dstar.k_m == 0
    assert dstar.rhs.shape == (5,5)
    assert dstar.g.shape == (5,5)
    assert dstar.rhs[dstar.s_goal] == 0
    assert np.isinf(dstar.g[dstar.s_goal])
    
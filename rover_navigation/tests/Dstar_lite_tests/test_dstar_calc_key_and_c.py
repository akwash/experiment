import math
from planning.dstar_lite import DStarLite
from mapping.grid_map import OccupancyGridMap, OBSTACLE, UNOCCUPIED

def test_calculate_key_mono():
    m = OccupancyGridMap(5,5)
    dstar = DStarLite(m, OccupancyGridMap, (0,0), (4,4))

    k_goal = dstar.calculate_key(dstar.s_goal)
    k_start = dstar.calculate_key(dstar.s_start)

    assert (k_goal.l1 <= k_start.k1) or (k_goal.k1 == k_start.k1 and k_goal.k2 <= k_start.k2)

def test_cost_free_and_obstacle():
    m = OccupancyGridMap(5,5)
    dstar = DStarLite(m, OccupancyGridMap, (0,0), (4,4))

    c_free = dstar.c((0,0), (3,4))
    assert c_free == math.sqrt(3**2 + 4**2)

    dstar.sensed_map.set_obstacle((1,1))
    c_blocked = dstar.c((0,0), (1,1))
    assert c_blocked == float('inf')

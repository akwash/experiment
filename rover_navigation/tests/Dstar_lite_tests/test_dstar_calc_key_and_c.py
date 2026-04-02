import math
from rover_navigation.planning.dstar_lite import DStarLite
from rover_navigation.mapping.occupancy_map import OccupancyMap, OBSTACLE, UNOCCUPIED

def test_calculate_key_mono():
    m = OccupancyMap(5,5)
    dstar = DStarLite(m, (0,0), (4,4))

    k_goal = dstar.calculate_key(dstar.s_goal)
    k_start = dstar.calculate_key(dstar.s_start)

    assert (k_goal.k1 <= k_start.k1) or (k_goal.k1 == k_start.k1 and k_goal.k2 <= k_start.k2)

def test_cost_free_and_obstacle():
    m = OccupancyMap(5,5)
    dstar = DStarLite(m, (0,0), (4,4))

    c_free = dstar.c((0,0), (3,4))
    assert c_free == math.sqrt(3**2 + 4**2)

    dstar.sensed_map.set_obstacles((1,1))
    c_blocked = dstar.c((0,0), (1,1))
    assert c_blocked == float('inf')

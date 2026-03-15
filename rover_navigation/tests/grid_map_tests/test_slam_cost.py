from mapping.grid_map import OccupancyGridMap, SLAM, OBSTACLE
import math

def test_cost_free():
    m = OccupancyGridMap(5,5)
    slam = SLAM(m, view_ranges = 1)

    assert slam.c((0,0), (3,4)) == math.sqrt(3 ** 2 + 4 ** 2)

def test_cost_obstacle():
    m = OccupancyGridMap(5,5)
    m.set_obstacle((1,1))
    slam = SLAM(m, view_range =1 )
    assert slam.c((1,1),(2,2)) == float('inf')

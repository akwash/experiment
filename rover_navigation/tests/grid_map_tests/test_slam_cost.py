from rover_navigation.mapping.occupancy_map import OccupancyMap, SLAM, OBSTACLE
import math

def test_cost_free():
    m = OccupancyMap(5,5)
    slam = SLAM(m, view_range=1)
    assert slam.cost((0,0), (3,4)) == math.sqrt(3**2 + 4**2)

def test_cost_obstacle():
    m = OccupancyMap(5,5)
    m.set_obstacles((1,1))   # correct method name
    slam = SLAM(m, view_range=1)
    assert slam.cost((1,1),(2,2)) == float('inf')
from mapping.occupancy_map import OccupancyMap, OBSTACLE

def test_succesors_4n():
    m = OccupancyMap(5,5,exploration_setting = '4N')
    succ = m.succesorr((2,2))
    assert set(succ) =={(3,2),(2,3),(1,2),(2,1)}

def test_succesors_8n():
    m = OccupancyMap(5,5,exploration_setting = '8N')
    succ = m.succesor((2,2))
    assert len(succ) == 8

def test_succesors_filters_obstacles():
    m = OccupancyMap(5,5)
    m.set_obstacle((3,2))
    succ = m.succesor((2,2), avoid_obstacles = True)
    assert (3,2) not in succ
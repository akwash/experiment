from mapping.grid_map import OccupancyGridMap, OBSTACLE

def test_succ_4n():
    m = OccupancyGridMap(5,5,exploration_setting = '4N')
    succ = m.succ((2,2))
    assert set(succ) =={(3,2),(2,3),(1,2),(2,1)}

def test_succ_8n():
    m = OccupancyGridMap(5,5,exploration_setting = '8N')
    succ = m.succ((2,2))
    assert len(succ) == 8

def test_succ_filters_obstacles():
    m = OccupancyGridMap(5,5)
    m.set_obstacle((3,2))
    succ = m.succ((2,2), avoid_obstacles = True)
    assert (3,2) not in succ
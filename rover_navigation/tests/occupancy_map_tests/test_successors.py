from rover_navigation.mapping.occupancy_map import OccupancyMap, OBSTACLE

def test_successors_4n():
    m = OccupancyMap(5,5,movement_setting = '4N')
    succ = m.successors((2,2))
    assert set(succ) =={(3,2),(2,3),(1,2),(2,1)}

def test_successors_8n():
    m = OccupancyMap(5,5,movement_setting = '8N')
    succ = m.successors((2,2))
    assert len(succ) == 8

def test_successors_filters_obstacles():
    m = OccupancyMap(5,5)
    m.set_obstacles((3,2))
    succ = m.successors((2,2), avoid_obstacles = True)
    assert (3,2) not in succ
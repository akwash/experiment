import math
from typing import List

# each vertex is a node on the grid
class Vertex:
    def __init__(self, pos: (int, int)):
        self.pos = pos # tuple (x,y)
        self.edges_and_costs = {} # neighbor psitions and movement cost

    # add directed edge from this point to another point with the cost
    def add_edge_with_cost(self, succ: (int, int), cost: float):
        if succ != self.pos:
            self.edges_and_costs[succ] = cost
    

# container for multiple points on the grid
class Vertices:
    def __init__(self):
        self.list = []

    # add vertex to the list
    def add_vertex(self, v: Vertex):
        self.list.append(v)

    @property 
    def vertices(self):
        return self.list

# euclidian distance between two points on the grid
def heuristic(p: (int, int), q: (int, int)) -> float:
    """
        helper function to compute distance between two points
        :param p: (x1,y1)
        :param q: (x2, y2)
        :return: euclidian distance
    """
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)

# return the 4 neighbors (right, up, left, down)
def get_movements_4n(x: int, y: int) -> List:
    """ 
    get the 4 connectivity movements
    :return: list of movements with associated cost, form [dx, dy, cost]
    """
    return [(x + 1, y + 0),
            (x + 0, y + 1),
            (x - 1, y + 0),
            (x + 0, y - 1)]

# return the 4 original neighbors and 4 diagonal neighbors, allows for diagonal movement
def get_movements_8n(x: int, y: int) -> List:
    """
    get possible 8 connectivity movements
    :return: list of movements with cost
    """
    return [(x + 1, y + 0),
            (x + 0, y + 1),
            (x - 1, y + 0),
            (x + 0, y - 1),
            (x + 1, y + 1),
            (x - 1, y + 1),
            (x - 1, y - 1),
            (x + 1, y - 1)]
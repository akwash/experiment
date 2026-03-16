import numpy as np
from util.utils import get_movements_4n, get_movements_8n, heuristic, Vertices, Vertex
from typing import Dict, List

# set values for occupied and unoccupied cells in the map
OBSTACLE = 255 
UNOCCUPIED = 0

class OccupancyMap:
    def __init__(self, x_dim, y_dim, movement_setting = '4N'):
        """
        sets intial values for the occupancy grid
        :param x_dim: dimension of the map in the x-direction
        :param y_dim: dimension of the map in the y-direction
        :param movement_setting: either '4N' or '8N', decides what movements are allowed
        """
        self.x_dim = x_dim
        self.y_dim = y_dim

        # map boundaries (units in m)
        self.map_boundaries = (x_dim, y_dim)

        # set occupancy map as unoccupied everywhere
        self.occupancy_map = np.zeros(self.map_boundaries,dtype=np.uint8)

        # set the obstacles (known by visiting cells)
        self.visited = {} # set to no visited cells
        self.movement_setting = movement_setting # decide how to traverse

    def get_map(self):
        """
        :return: return the current occupancy map
        """
        return self.occupancy_map
    
    def set_map(self, new_omap):
        """
        :param new_omap: new occupancy map
        :return: None
        """
        self.occupancy_map = new_omap

    def in_bounds(self, cell: (int, int)) -> bool:
        """
        check if the cell is within the map boundaries
        :param cell: cell position to check (x,y)
        :return: true if within, false if outside
        """
        (x,y) = cell 
        return 0 <= x < self.x_dim and 0 <= y < self.y_dim
    
    def is_unoccupied(self, pos: (int, int)) -> bool:
        """
        check if a cell is unoccupied
        :param pos: cell position to check (x,y)
        :return: true if the cell is unoccupied, false if the cell is occupied
        """
        (x,y) = (round(pos[0]), round(pos[1])) # round to the closest (x,y) pos

        (row,col) = (x,y)

        # check if the cell is out of bounds
        if not self.in_bounds(cell=(x,y)):
            raise IndexError("Map index out of bounds")
        
        return self.occupancy_map[row][col] == UNOCCUPIED
    
    def filter(self, neighbors: List, avoid_obstacles: bool):
        """
        filter out the neighbors that are out of bounds/occupied
        :param neighbors: list of neighbor cells
        :param avoid_obstacles: if true filter out occupied cells
        """
        if avoid_obstacles:
            return [node for node in neighbors if self.in_bounds(node) and self.is_unoccupied(node)] # filter out out of bounds and occupied
        
        return [node for node in neighbors if self.in_bounds(node)] # filter out out of bounds
    
    def succesors(self, vertex: (int, int), avoid_obstacles: bool = False) -> list:
        """
        get the successors of a cell (sucessors are neighbors cells that can be reached)
        :param vertex: cell to find successors for 
        :param avoid_obstacles: if true, filter out occupied cells
        :return: list of successors
        """
        (x,y) = vertex

        if self.movement_setting == '4N':
            # move up, down, left, right
            movements = get_movements_4n(x =x, y=y)
        else: 
            movements = get_movements_8n(x=x, y=y)
            # move up, down, left, right, and diagonals

        if (x+y) % 2 == 0: movements.reverse() 

        filtered_movements = self.filter(neighbors = movements, avoid_obstacles=avoid_obstacles) # filter out movements that cause out of bounds and occupied cells
        return list(filtered_movements)
    
    def set_obstacles(self, pos: (int,int)):
        """
        :param pos: cell position to set as an obstacle (x,y)
        :return: None
        """
        (x,y) = round(pos[0]), round(pos[1]) # round to nearest match
        (row, col) = (x,y)

        self.occupancy_map[row][col] = OBSTACLE # set cell as an obstacle

    def remove_obstacle(self, pos: (int,int)):
        """
        :param pos: cell position to set as an obstacle (x,y)
        :return: None
        """
        (x,y) = round(pos[0]), round(pos[1]) # round to nearest match
        (row, col) = (x,y)

        self.occupancy_map[row][col] = UNOCCUPIED # set cell as unoccupied

    def observations(self, global_pos: (int, int), view_range = int =2 ) -> Dict:
        """
        :param global_pos: current global position of system (x,y)
        :param view_range: how far can the system see in each direction
        :return: dictionary of observations [(x,y): occupancy val]
        """
        (px,py) = global_pos

        nodes = [(x, y) for x in range(px - view_range, px + view_range + 1)
                 for y in range(py - view_range, py + view_range + 1)
                 if self.in_bounds((x, y))]

        return {node: UNOCCUPIED if self.is_unoccupied(pos=node) else OBSTACLE for node in nodes}
    
class SLAM:
    def __init__(self, map: OccupancyMap, view_range: int):
        self.truth_map = map # true map of environment
        self.slam_map = OccupancyMap(x_dim=map.x_dim, y_dim=map.y_dim)

        self.slam_map.occupancy_map = map.occupancy_map.copy()

        self.view_range = view_range

    def set_ground_map(self, g_map: OccupancyMap):
        """
        set the ground truth map for SLAM
        :param: g_map: new ground truth map
        """
        self.set_ground_map = g_map

    def cost(self, u: (int, int), v: (int, int)) -> float:
        """
        calculate the cost between two nodes
        :param u: from vertex
        :param v: to vertex
        """

        if not self.slam_map.is_unoccupied(u) or not self.slam_map.is_unoccupied(v):
            return float('inf') # if either node is occupied, cost is infinite
        else: 
            return heuristic(u,v) # otherwise, cost is heuristic distance
        
    def rescan(self, global_pos: (int, int)):
        """
        rescan the area around curr position and update SLAM
        :param global_pos: current global position (x,y)
        """

        local_observation = self.truth_map.observations(global_pos=global_po, view_range = self.view_range) # get local observations
        
        vertices = self.update_changed_edge_costs(local_grid = observations) # update the SLAM map with new observations and get the vertices that changed from unoccupied to occupied or vice versa
        return vertices, self.slam_map
    
    def update_changed_edge_costs(self, local_grid: Dict) -> Vertices:
        """
        update SLAM map with new observations and get changed vertices
        :param local_grid: new local observations
        :param vertices: list of vertices that changed
        :return: list of vertices that changed
        """
        vertices = Vertices()
        
        for node, value in local_grid.items():
            # if there is an obstacle in new observations
            if value == OBSTACLE:
                if self.slam_map.is_unoccupied(node):
                    v = Vertex(pos=node)
                    succ = self.slam_map.successor(node)
                    for u in succ:
                        v.add_edge_with_cost(succ=u, cost=self.cost(u,v.pos))
                    vertices.add_vertex(v)
                    self.slam_map.set_obstacles(node)
                else:
                    if not self.slam_map.is_unoccupied(node):
                        v = Vertex(pos=node)
                        succ = self.slam_map.successor(node)
                        for u in succ:
                            v.add_edge_with_cost(succ=u, cost=self.cost(u,v.pos))
                        vertices.add_vertex(v)
                        self.slam_map.remove_obstacle(node)

        return vertices
    
        
    

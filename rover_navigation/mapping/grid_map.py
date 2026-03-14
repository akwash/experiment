import numpy as np

class OccupancyGridMap:
    def __init(self, x_dim, y_dim, exploration_setting='8N'):
        """
        set intial vals for the occupancy grid
        :param x_dim: dimension in x-direction
        :param y_dim: dimension in y-direction
        """
        # these are in meters
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.map_extents = (x_dim, y_dim)

        # 0 is unoccupied, 1 is occupied, 2 is unknown
        self.occupancy_grid_map = np.zeros(self.map_extents, dtype=np.uint8)

        # obstacles
        self.visited = {}
        self.exploration_setting = exploration_setting

        # get occupancy map
        def get_map(self):
            """
            : return: current occpancy of map
            """
            return self.occupancy_grid_map
        
        # set occupancy map
        def set_map(self,new_ogrid):
            """
            :param new_ogrid: new occupancy grid to set
            :return: none
            """
            self.occupancy_grid_map = new_ogrid
        
        # check if cell is unoccupied
        def is_unoccupied(self, pos: (int,int)) -> bool:
            """
            :param pos: cell position to be checked
            :return: true if occupied, false if free
            """
            (x,y) = (round(pos[0]), round(pos[1]))
            (row, col) = (x,y)

            return self.occupancy_grid_map[row][col] == 0
        
        # check if cell is within the bounds
        def in_bounds(self,cell: (int,int)) -> bool:
            """
            check if the coords are within the bounds fo the  grid map.
            :param cell: cell position (x,y)
            :return: true if within, false if outside
            """
            (x,y) = cell
            return 0 <= x < self.x_dim and 0 <= y < self.y_dim
        
        # 
        def filter(self, neighbors: List, avoid_obstacles: bool):
            """
            :param neighbors: list of potential neighbors before filtering
            :param avoid_obstacles: if true, filter out obstacle cells
            :return:
            """
            if avoid_obstacles:
                return [node for node in neighbors if self.in_bounds(node) and self.is_unoccupied(node)]
            return [node for node in neighbors if self.in_bounds(node)]

        # 
        def succ(self, vertex: (int, int), avoid_obstacles: bool = False) -> list:
        """
        :param avoid_obstacles:
        :param vertex: vertex you want to find direct successors from
        :return:
        """
        (x, y) = vertex

        if self.exploration_setting == '4N':  # change this
            movements = get_movements_4n(x=x, y=y)
        else:
            movements = get_movements_8n(x=x, y=y)

        # not needed. Just makes aesthetics to the path
        if (x + y) % 2 == 0: movements.reverse()

        filtered_movements = self.filter(neighbors=movements, avoid_obstacles=avoid_obstacles)
        return list(filtered_movements)

    def set_obstacle(self, pos: (int, int)):
        """
        :param pos: cell position we wish to set obstacle
        :return: None
        """
        (x, y) = (round(pos[0]), round(pos[1]))  # make sure pos is int
        (row, col) = (x, y)
        self.occupancy_grid_map[row, col] = OBSTACLE

    def remove_obstacle(self, pos: (int, int)):
        """
        :param pos: position of obstacle
        :return: None
        """
        (x, y) = (round(pos[0]), round(pos[1]))  # make sure pos is int
        (row, col) = (x, y)
        self.occupancy_grid_map[row, col] = UNOCCUPIED

    def local_observation(self, global_position: (int, int), view_range: int = 2) -> Dict:
        """
        :param global_position: position of robot in the global map frame
        :param view_range: how far ahead we should look
        :return: dictionary of new observations
        """
        (px, py) = global_position
        nodes = [(x, y) for x in range(px - view_range, px + view_range + 1)
                 for y in range(py - view_range, py + view_range + 1)
                 if self.in_bounds((x, y))]
        return {node: UNOCCUPIED if self.is_unoccupied(pos=node) else OBSTACLE for node in nodes}


class SLAM:
    def __init__(self, map: OccupancyGridMap, view_range: int):
        self.ground_truth_map = map
        self.slam_map = OccupancyGridMap(x_dim=map.x_dim,
                                         y_dim=map.y_dim)
        self.view_range = view_range

    def set_ground_truth_map(self, gt_map: OccupancyGridMap):
        self.ground_truth_map = gt_map

    def c(self, u: (int, int), v: (int, int)) -> float:
        """
        calcuclate the cost between nodes
        :param u: from vertex
        :param v: to vertex
        :return: euclidean distance to traverse. inf if obstacle in path
        """
        if not self.slam_map.is_unoccupied(u) or not self.slam_map.is_unoccupied(v):
            return float('inf')
        else:
            return heuristic(u, v)

    def rescan(self, global_position: (int, int)):

        # rescan local area
        local_observation = self.ground_truth_map.local_observation(global_position=global_position,
                                                                    view_range=self.view_range)

        vertices = self.update_changed_edge_costs(local_grid=local_observation)
        return vertices, self.slam_map

    def update_changed_edge_costs(self, local_grid: Dict) -> Vertices:
        vertices = Vertices()
        for node, value in local_grid.items():
            # if obstacle
            if value == OBSTACLE:
                if self.slam_map.is_unoccupied(node):
                    v = Vertex(pos=node)
                    succ = self.slam_map.succ(node)
                    for u in succ:
                        v.add_edge_with_cost(succ=u, cost=self.c(u, v.pos))
                    vertices.add_vertex(v)
                    self.slam_map.set_obstacle(node)
            else:
                # if white cell
                if not self.slam_map.is_unoccupied(node):
                    v = Vertex(pos=node)
                    succ = self.slam_map.succ(node)
                    for u in succ:
                        v.add_edge_with_cost(succ=u, cost=self.c(u, v.pos))
                    vertices.add_vertex(v)
                    self.slam_map.remove_obstacle(node)
        return vertices

        


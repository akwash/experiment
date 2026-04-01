import numpy as np
from rover_navigation.util.utils import get_movements_4n, get_movements_8n, heuristic, Vertices, Vertex
from typing import Dict, List
import matplotlib.pyplot as plt


# set values for occupied and unoccupied cells in the map
OBSTACLE = 1
UNOCCUPIED = 0

class OccupancyMap:
    def __init__(self, x_dim, y_dim, movement_setting = '8N'):
        """
        sets intial values for the occupancy grid
        :param x_dim: dimension of the map in the x-direction
        :param y_dim: dimension of the map in the y-direction
        :param movement_setting: either '4N' or '8N', decides what movements are allowed
        """
        self.x_dim = x_dim
        self.y_dim = y_dim

        # map boundaries (units in m)
        self.map_boundaries = (y_dim, x_dim)

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

    cell = tuple[int, int]

    def in_bounds(self, cell) -> bool:
        """
        check if the cell is within the map boundaries
        :param cell: cell position to check (x,y)
        :return: true if within, false if outside
        """
        (x,y) = cell 
        return 0 <= x < self.x_dim and 0 <= y < self.y_dim
    
    pos = tuple[int, int]

    def is_unoccupied(self, pos) -> bool:
        """
        check if a cell is unoccupied
        :param pos: cell position to check (x,y)
        :return: true if the cell is unoccupied, false if the cell is occupied
        """
        (x,y) = (round(pos[0]), round(pos[1])) # round to the closest (x,y) pos

        (row,col) = (y,x)

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
    
    vertex = tuple[int, int]

    def successors(self, vertex, avoid_obstacles: bool = False) -> list:
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
    
    def set_obstacles(self, pos):
        """
        :param pos: cell position to set as an obstacle (x,y)
        :return: None
        """
        (x,y) = round(pos[0]), round(pos[1]) # round to nearest match
        (row, col) = (y,x)

        self.occupancy_map[row][col] = OBSTACLE # set cell as an obstacle

    def remove_obstacle(self, pos):
        """
        :param pos: cell position to set as an obstacle (x,y)
        :return: None
        """
        (x,y) = round(pos[0]), round(pos[1]) # round to nearest match
        (row, col) = (y,x)

        self.occupancy_map[row][col] = UNOCCUPIED # set cell as unoccupied

    def observations(self, global_pos: cell, view_range: int = 2) -> Dict:
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
    def inflate(self, radius: int = 1):
        """
        Inflate obstacles to account for rover size / safety margin
        :param radius: number of grid cells to expand obstacles
        """
        grid = self.occupancy_map
        h, w = grid.shape

        new_grid = grid.copy()

        obstacle_cells = np.argwhere(grid == OBSTACLE)

        for row, col in obstacle_cells:
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    rr = row + dr
                    cc = col + dc

                    if 0 <= rr < h and 0 <= cc < w:
                        new_grid[rr, cc] = OBSTACLE

        self.occupancy_map = new_grid
    
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
        self.truth_map = g_map

    u = tuple[int, int]
    v = tuple[int, int]

    def cost(self, u, v) -> float:
        """
        calculate the cost between two nodes
        :param u: from vertex
        :param v: to vertex
        """

        if not self.slam_map.is_unoccupied(u) or not self.slam_map.is_unoccupied(v):
            return float('inf') # if either node is occupied, cost is infinite
        else: 
            return heuristic(u,v) # otherwise, cost is heuristic distance
        
    global_pos = tuple[int, int]

    def rescan(self, global_pos):
        """
        rescan the area around curr position and update SLAM
        :param global_pos: current global position (x,y)
        """

        local_observation = self.truth_map.observations(global_pos=global_pos, view_range = self.view_range) # get local observations
        
        vertices = self.update_changed_edge_costs(local_grid = local_observation) # update the SLAM map with new observations and get the vertices that changed from unoccupied to occupied or vice versa
        return vertices, self.slam_map
    
    def update_changed_edge_costs(self, local_grid: Dict) -> Vertices:
        vertices = Vertices()

        for node, value in local_grid.items():
            if value == OBSTACLE:
                if self.slam_map.is_unoccupied(node):
                    v = Vertex(pos=node)
                    succ = self.slam_map.successors(node)
                    for u in succ:
                        v.add_edge_with_cost(succ=u, cost=self.cost(u, v.pos))
                    vertices.add_vertex(v)
                    self.slam_map.set_obstacles(node)
            else:
                if not self.slam_map.is_unoccupied(node):
                    v = Vertex(pos=node)
                    succ = self.slam_map.successors(node)
                    for u in succ:
                        v.add_edge_with_cost(succ=u, cost=self.cost(u, v.pos))
                    vertices.add_vertex(v)
                    self.slam_map.remove_obstacle(node)

        return vertices
    
def build_map_from_predictions(
            points:np.ndarray,
            pred_labels: np.ndarray,
            grid_resolution:float = 0.1524, # m per cell
            obstacle_label: int = 1,
    ):
        """ 
        Conver the RandLA-Net predictions to an Occupancy map
        :param points: (N,3), xyz
        :param pred_labels: (N,)
        :param grid_resolution: meters per cell
        :param obstacle_label: obstacle class

        :return occupancy_map: OccupancyMap instance
        :return grid_info: dictionary for coordinate conversion
        """
        # global enviornment boundaries:
            # 0,0 at bottom left cell
            # y_min = 0, x_min = 0
            # y_max = 15 ft, x_max = 15 ft

        workspace_size_m = 15 * 0.3048 # 15ft (value in m)
       
        min_x = 0
        min_y = 0
        max_x = workspace_size_m
        max_y = workspace_size_m

        
        #mm
        # filter out ceilings
        min_z = -5
        max_z = 30

        # crop to the working area
        mask = (
            (points[:,0] >= min_x) & (points[:,0]<=max_x) &
            (points[:,1] >= min_y) & (points[:,1]<=max_y) &
            (points[:,2] >= min_z) & (points[:,2]<=max_z))
        
        points = points[mask]
        pred_labels = pred_labels[mask]

        # project to 2D plane (x,y)
        xy = points[:,:2]


        width = int((max_x - min_x) / grid_resolution)
        height = int((max_y - min_y) / grid_resolution)

        omap = OccupancyMap(width, height)

        # get the obstacle points
        obstacle_points = xy[pred_labels == obstacle_label]

        # convert to the grid indices
        cols = ((obstacle_points[:,0] - min_x) / grid_resolution).astype(int)
        rows = ((obstacle_points[:,1] - min_y) / grid_resolution).astype(int)

        # bounds
        valid = (
            (rows >= 0) & (rows < height) &
            (cols >= 0) & (cols < width)
        )

        rows = rows[valid]
        cols = cols[valid]

        # fill the map
        counts = {}

        for r, c in zip(rows, cols):
            counts[(r, c)] = counts.get((r, c), 0) + 1

        for (row, col), count in counts.items():
            if count >= 2:
                omap.set_obstacles((col, row))

        grid_info = {
            "resolution": grid_resolution,
            "min_x": min_x,
            "min_y": min_y
        }

        return omap, grid_info
    
def world_to_grid(x: float, y: float, grid_info: dict) -> tuple[int, int]:
    col = int((x - grid_info["min_x"]) / grid_info["resolution"])
    row = int((y - grid_info["min_y"]) / grid_info["resolution"])
    return row, col


def grid_to_world(row: int, col: int, grid_info: dict) -> tuple[float, float]:
    x = grid_info["min_x"] + col * grid_info["resolution"]
    y = grid_info["min_y"] + row * grid_info["resolution"]
    return x, y

def visualize_empty_grid(omap):
    grid = omap.get_map()
    h, w = grid.shape

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6,6))
    plt.pcolormesh(grid, edgecolors='black', linewidth=0.3, cmap='gray_r')

    ax = plt.gca()
    ax.set_aspect('equal')
    

    plt.title("Grid")
    plt.show()

def transform_to_world(points, rover_pose_xy, heading_rad):
    """
    Transform local sensor frame points into world frame.
    points: (N,3) xyz in local sensor frame
    :param rover_pose_xy: (x, y) world position of rover in meters
    :paramheading_rad: rover yaw angle in radians
    :return: (N,3) points in world frame
    """
    cos_h = np.cos(heading_rad) # rot matrix
    sin_h = np.sin(heading_rad) # rot matrix

    R = np.array([[cos_h, -sin_h],
                  [sin_h,  cos_h]]) # yaw rotation matrix

    xy_local = points[:, :2] # local xy coords
    xy_world = (R @ xy_local.T).T + np.array(rover_pose_xy) # world xy coords after rot and transl

    points_world = points.copy() # change xy to world
    points_world[:, :2] = xy_world # update xy to world
    return points_world

if __name__ == "__main__":
    # create a test map
    omap = OccupancyMap(30, 30)

    # visualize it
    visualize_empty_grid(omap)


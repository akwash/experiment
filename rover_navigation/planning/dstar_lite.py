import numpy as np
from planning.priority_queue import PriorityQueue, Priority
from mapping.occupancy_map import OccupancydMap
from util.utils import heuristic, Vertex, Vertices
from typing import Dict, List

class DStarLite:
    
    def __init__(self, map, OccupancyGridMap, s_start:(int, int),s_goal:(int, int)):
        """
        :param map: ground truth map of environment
        :param s_start: start location
        :param s_goal: end location
        """
        self.new_edges_and_old_costs = None; # store new edges and old costs for updates

        # algorithm:
        self.s_start = s_start # start loc
        self.s_goal = s_goal # goal loc
        self.s_last = s_start # last loc
        self.k_m = 0 # accumative heuristic change
        self.U = PriorityQueue() # priority queue for states to be processed
        self.rhs= np.ones((map.x_dim,map.y_dim))*np.inf # rhs values for all states
        self.g = self.rhs.copy() # g values for all states

        # map:
        self.sensed_map = OccupancyGridMap(x_dim=map.x_dim,
                                            y_dim=map.y_dim,
                                            exploration_setting='8N')
        
        self.rhs[self.s_goal] = 0 # rhs of goal is 0
        self.U.insert(self.s_goal, Priority(heuristic(self.s_start, self.s_goal), 0)) # goal into priority queue w/ heuristic as priority
    
    # heuristic function for calculating priority
    # input: two vertices
    # output: euclidian distance between input vertices
    def calculate_key(self,s: (int, int)):
        """
        :param s: vertex to calculate key
        :return: priority class of two keys
        """
        k1 = min(self.g[s],self.rhs[s])+heuristic(self.s_start,s)+self.k_m # heuristic + accumulative heuristic change + min(g, rhs)
        k2 = min(self.g[s],self.rhs[s]) # min(g, rhs)
        return Priority(k1,k2)

    # cost function for edge costs
    # input: two vertices
    # output: euclidian distance between input vertices if both are unoccupied, inf if occupied
    def c(self, u: (int,int),v: (int,int)) -> float:
        """ calcualte the cost between nodes
        :param u: from vertex
        :param v: to vertex
        return: euclidian distance to reverse. inf if obstacle is in the path
        """
        if not self.sensed_map.is_unoccupied(u) or not self.sensed_map.is_unoccupied(v):
            return float('inf')
        else:
            return heuristic(u,v)
        
    # check if the vertex is in priority queue
    # input: vertex
    # output: booleean
    def contain(self,u:(int,int)) -> (int,int):
        return u in self.U.vertices_in_heap
    
    # update vertex in priority queue
    # input: vertex
    def update_vertex(self,u:(int,int)):
        if self.g[u] != self.rhs[u] and self.contain(u):
            self.U.update(u,self.calculate_key(u))
        elif self.g[u] != self.rhs[u] and not self.contain(u):
            self.U.insert(u, self.calculate_key(u))
        elif self.g[u] == self.rhs[u] and self.contain(u):
            self.U.remove(u)

    def compute_shortest_path(self):
        while self.U.top_key() < self.calculate_key(self.s_start) or self.rhs[self.s_start] > self.g[self.s_start]:
            u = self.U.top()
            k_old = self.U.top_key()
            k_new = self.calculate_key(u)

            if k_old < k_new:
                self.U.update(u, k_new)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                self.U.remove(u)
                pred = self.sensed_map.succ(vertex=u)
                for s in pred:
                    if s != self.s_goal:
                        self.rhs[s] = min(self.rhs[s], self.c(s, u) + self.g[u])
                    self.update_vertex(s)
            else:
                self.g_old = self.g[u]
                self.g[u] = float('inf')
                pred = self.sensed_map.succ(vertex=u)
                pred.append(u)
                for s in pred:
                    if self.rhs[s] == self.c(s, u) + self.g_old:
                        if s != self.s_goal:
                            min_s = float('inf')
                            succ = self.sensed_map.succ(vertex=s)
                            for s_ in succ:
                                temp = self.c(s, s_) + self.g[s_]
                                if min_s > temp:
                                    min_s = temp
                            self.rhs[s] = min_s
                    self.update_vertex(u)
    
    # rescan the graph for changes and store
    def rescan(self) -> Vertices:
        new_edges_and_old_costs = self.new_edges_and_old_costs
        self.new_edges_and_old_costs = None
        return new_edges_and_old_costs
    
    # move and replan
    # input: current position
    # output: path from start to goal, g vals, rhs vals
    def move_and_replan(self, robot_position: (int, int)):
        path = [robot_position]
        self.s_start = robot_position
        self.s_last = self.s_start
        self.compute_shortest_path()

        # move along path until reaching goal
        while self.s_start != self.s_goal:
            assert (self.rhs[self.s_start] != float('inf')), "There is no known path!"

            succ = self.sensed_map.succ(self.s_start, avoid_obstacles=False) # get successors of cur pos
            min_s = float('inf') 
            arg_min = None

            # find successor with lowest g + c
            for s_ in succ:
                temp = self.c(self.s_start, s_) + self.g[s_]
                if temp < min_s:
                    min_s = temp
                    arg_min = s_

            ### algorithm sometimes gets stuck here for some reason !!! FIX
            self.s_start = arg_min
            path.append(self.s_start)
            # scan graph for changed costs
            changed_edges_with_old_cost = self.rescan()
            #print("len path: {}".format(len(path)))
            # if any edge costs changed
            if changed_edges_with_old_cost:
                self.k_m += heuristic(self.s_last, self.s_start)
                self.s_last = self.s_start

                # for all directed edges (u,v) with changed edge costs
                vertices = changed_edges_with_old_cost.vertices
                for vertex in vertices:
                    v = vertex.pos
                    succ_v = vertex.edges_and_c_old
                    for u, c_old in succ_v.items():
                        c_new = self.c(u, v)
                        if c_old > c_new:
                            if u != self.s_goal:
                                self.rhs[u] = min(self.rhs[u], self.c(u, v) + self.g[v])
                        elif self.rhs[u] == c_old + self.g[v]:
                            if u != self.s_goal:
                                min_s = float('inf')
                                succ_u = self.sensed_map.succ(vertex=u)
                                for s_ in succ_u:
                                    temp = self.c(u, s_) + self.g[s_]
                                    if min_s > temp:
                                        min_s = temp
                                self.rhs[u] = min_s
                            self.update_vertex(u)
            self.compute_shortest_path()
        print("path found!")
        return path, self.g, self.rhs

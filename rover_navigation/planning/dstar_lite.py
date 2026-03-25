import numpy as np
from rover_navigation.planning.priority_queue import PriorityQueue, Priority
from rover_navigation.mapping.occupancy_map import OccupancyMap
from rover_navigation.util.utils import heuristic, Vertices


class DStarLite:
    def __init__(self, map: OccupancyMap, s_start: tuple[int, int], s_goal: tuple[int, int]):
        """
        :param map: occupancy map of environment
        :param s_start: start location
        :param s_goal: goal location
        """
        self.new_edges_and_old_costs = None

        # algorithm
        self.s_start = s_start
        self.s_goal = s_goal
        self.s_last = s_start
        self.k_m = 0
        self.U = PriorityQueue()

        # rhs/g arrays indexed as [y,x] (rows, col)
        self.rhs = np.full(map.get_map().shape, np.inf)
        self.g = self.rhs.copy()

        # map
        self.sensed_map = OccupancyMap(
            x_dim=map.x_dim,
            y_dim=map.y_dim,
            movement_setting=map.movement_setting,
        )
        self.sensed_map.set_map(map.get_map().copy())

        self.rhs[self.s_goal] = 0
        self.U.insert(self.s_goal, Priority(heuristic(self.s_start, self.s_goal), 0))

    def calculate_key(self, s: tuple[int, int]) -> Priority:
        """
        :param s: vertex to calculate key
        :return: priority class of two keys
        """
        k1 = min(self.g[s], self.rhs[s]) + heuristic(self.s_start, s) + self.k_m
        k2 = min(self.g[s], self.rhs[s])
        return Priority(k1, k2)

    def c(self, u: tuple[int, int], v: tuple[int, int]) -> float:
        """
        calculate the cost between nodes
        :param u: from vertex
        :param v: to vertex
        :return: euclidean distance if both are free, inf if obstacle blocks path
        """
        if not self.sensed_map.is_unoccupied(u) or not self.sensed_map.is_unoccupied(v):
            return float("inf")
        return heuristic(u, v)

    def contain(self, u: tuple[int, int]) -> bool:
        return u in self.U.vertices_in_heap

    def update_vertex(self, u: tuple[int, int]) -> None:
        if self.g[u] != self.rhs[u] and self.contain(u):
            self.U.update(u, self.calculate_key(u))
        elif self.g[u] != self.rhs[u] and not self.contain(u):
            self.U.insert(u, self.calculate_key(u))
        elif self.g[u] == self.rhs[u] and self.contain(u):
            self.U.remove(u)

    def compute_shortest_path(self) -> None:
        while self.U.top_key() < self.calculate_key(self.s_start) or self.rhs[self.s_start] > self.g[self.s_start]:
            u = self.U.top()
            k_old = self.U.top_key()
            k_new = self.calculate_key(u)

            if k_old < k_new:
                self.U.update(u, k_new)

            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                self.U.remove(u)

                pred = self.sensed_map.successors(vertex=u, avoid_obstacles=False)
                for s in pred:
                    if s != self.s_goal:
                        self.rhs[s] = min(self.rhs[s], self.c(s, u) + self.g[u])
                    self.update_vertex(s)

            else:
                g_old = self.g[u]
                self.g[u] = float("inf")

                pred = self.sensed_map.successors(vertex=u, avoid_obstacles=False)
                pred.append(u)

                for s in pred:
                    if self.rhs[s] == self.c(s, u) + g_old:
                        if s != self.s_goal:
                            min_s = float("inf")
                            succ = self.sensed_map.successors(vertex=s, avoid_obstacles=False)
                            for s_ in succ:
                                temp = self.c(s, s_) + self.g[s_]
                                if min_s > temp:
                                    min_s = temp
                            self.rhs[s] = min_s
                    self.update_vertex(s)

    def rescan(self) -> Vertices | None:
        changed = self.new_edges_and_old_costs
        self.new_edges_and_old_costs = None
        return changed

    def update_map(self, new_map: OccupancyMap) -> None:
        """
        Replace sensed map from latest perception / mapping result.
        """
        self.sensed_map = OccupancyMap(
            x_dim=new_map.x_dim,
            y_dim=new_map.y_dim,
            movement_setting=new_map.movement_setting,
        )
        self.sensed_map.set_map(new_map.get_map().copy())

    def move_and_replan(self, robot_position: tuple[int, int]):
        path = [robot_position]
        self.s_start = robot_position
        self.s_last = self.s_start
        self.compute_shortest_path()

        while self.s_start != self.s_goal:
            assert self.rhs[self.s_start] != float("inf"), "There is no known path!"

            succ = self.sensed_map.successors(self.s_start, avoid_obstacles=True)
            min_s = float("inf")
            arg_min = None

            for s_ in succ:
                temp = self.c(self.s_start, s_) + self.g[s_]
                if temp < min_s:
                    min_s = temp
                    arg_min = s_

            if arg_min is None:
                raise RuntimeError("Planner got stuck: no valid successor found.")

            self.s_start = arg_min
            path.append(self.s_start)

            changed_edges_with_old_cost = self.rescan()

            if changed_edges_with_old_cost:
                self.k_m += heuristic(self.s_last, self.s_start)
                self.s_last = self.s_start

                vertices = changed_edges_with_old_cost.vertices
                for vertex in vertices:
                    v = vertex.pos
                    succ_v = vertex.edges_and_costs

                    for u, c_old in succ_v.items():
                        c_new = self.c(u, v)

                        if c_old > c_new:
                            if u != self.s_goal:
                                self.rhs[u] = min(self.rhs[u], self.c(u, v) + self.g[v])

                        elif self.rhs[u] == c_old + self.g[v]:
                            if u != self.s_goal:
                                min_cost = float("inf")
                                succ_u = self.sensed_map.successors(vertex=u, avoid_obstacles=False)
                                for s_ in succ_u:
                                    temp = self.c(u, s_) + self.g[s_]
                                    if min_cost > temp:
                                        min_cost = temp
                                self.rhs[u] = min_cost

                        self.update_vertex(u)

            self.compute_shortest_path()

        print("path found!")
        return path, self.g, self.rhs
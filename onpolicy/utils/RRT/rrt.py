"""

Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)

author: AtsushiSakai(@Atsushi_twi)

"""

import math
import random
from sys import float_repr_style

import matplotlib.pyplot as plt
import numpy as np

show_animation = False


class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    def __init__(self,
                 start,
                 goals,
                 obstacle_list,
                 rand_area,
                 expand_dis=3.0,
                 path_resolution=0.5,
                 goal_sample_rate=5,
                 max_iter=500):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:list of goals [[x,y], ...]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]

        """
        self.start = self.Node(start[0], start[1])
        self.end = [self.Node(x, y) for x,y in goals]
        self.num_goals = len(goals)
        self.min_rand_x = rand_area[0][0]
        self.max_rand_x = rand_area[0][1]
        self.min_rand_y = rand_area[1][0]
        self.max_rand_y = rand_area[1][1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []

    def planning(self, animation=True):
        """
        rrt path planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            if i % 100 == 0:
                print("Iter : %d, Nodes : %d"%(i, len(self.node_list)))
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node, self.obstacle_list):
                # self.node_list.append(new_node)
                self.push_node(new_node)

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].x,
                                      self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.get_nearest_goal(self.node_list[-1].x, self.node_list[-1].y),
                                        self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list):
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 5:
                self.draw_graph(rnd_node)

        return None  # cannot find path
    
    def select_frontiers(self, map, num_targets = 100):
        H, W = map.shape
        targets = []
        i = 0
        self.node_list = [self.start]
        while i < self.max_iter and len(targets) < num_targets:
            i += 1
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            x, y = int(new_node.x), int(new_node.y)
            x = max(0, min(x, H-1))
            y = max(0, min(y, W-1))
            if self.check_collision(new_node, self.obstacle_list):
                if map[x,y] == 1:
                    # unexplored
                    targets.append((x,y))
                else:
                    self.push_node(new_node)
        # print("%d iterations, %d targets"%(i, len(targets)))
        return targets

    def steer(self, from_node, to_node, extend_length=float("inf")):

        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length >= d:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
        else:
            new_node.x += extend_length * math.cos(theta)
            new_node.y += extend_length * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        # if extend_length > d:
        #    extend_length = d

        # n_expand = math.floor(extend_length / self.path_resolution)

        #for _ in range(n_expand):
        #    new_node.x += self.path_resolution * math.cos(theta)
        #    new_node.y += self.path_resolution * math.sin(theta)
        #    new_node.path_x.append(new_node.x)
        #    new_node.path_y.append(new_node.y)

        #d, _ = self.calc_distance_and_angle(new_node, to_node)
        #if d <= self.path_resolution:
        #    new_node.path_x.append(to_node.x)
        #    new_node.path_y.append(to_node.y)
        #    new_node.x = to_node.x
        #    new_node.y = to_node.y

        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind, ignore_goal = False):
        node = self.node_list[goal_ind]
        goal = self.get_nearest_goal(node.x, node.y)
        if ignore_goal:
            path = []
        else:
            path = [[goal.x, goal.y]]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        path = [x for x in reversed(path)]
        return path

    def calc_dist_to_goal(self, x, y):
        node = self.get_nearest_goal(x, y)
        dx = x - node.x
        dy = y - node.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.min_rand_x, self.max_rand_x),
                random.uniform(self.min_rand_y, self.max_rand_y))
        else:  # goal point sampling
            idx = random.randint(0, self.num_goals - 1)
            rnd = self.Node(self.end[idx].x, self.end[idx].y)
        return rnd
    
    def push_node(self, node):
        for p in self.node_list:
            if math.hypot(node.x - p.x, node.y - p.y) < 1e-5:
                return False
        self.node_list.append(node)
        return True

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        # print("%d nodes"%len(self.node_list))
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for obj in self.obstacle_list:
            if len(obj) == 3:
                ox, oy, size = obj
                print(obj)
                self.plot_circle(ox, oy, size)
            if len(obj) == 4:
                x1, y1, x2, y2 = obj
                self.plot_rectangle(x1, y1, x2, y2)

        plt.plot(self.start.x, self.start.y, "xr")
        for goal in self.end:
            plt.plot(goal.x, goal.y, "xr")
        plt.axis("equal")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.pause(0.01)
    
    def get_nearest_goal(self, x, y):
        dist = [math.hypot(n.x - x, n.y - y) for n in self.end]
        return self.end[dist.index(min(dist))]
    
    @staticmethod
    def plot_rectangle(x1, y1, x2, y2, color = "-b"):
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1 , y2-y1))

    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_collision(node, obstacleList, last_node = None):

        if node is None:
            return False
        
        if last_node == None:
            if node.parent is not None:
                last_node = node.parent
            else:
                last_node = None
        for obj in obstacleList:
            if len(obj) == 3:
                # cycle
                ox, oy, size = obj
                dx_list = [ox - x for x in node.path_x]
                dy_list = [oy - y for y in node.path_y]
                d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

                if min(d_list) <= size**2:
                    return False  # collision
            if len(obj) == 4:
                # rectangle
                x1, y1, x2, y2 = obj
                if node.x>=x1 and node.x<=x2 and node.y>=y1 and node.y<=y2:
                    # inside
                    return False
                if last_node == None:
                    continue
                corners = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
                for i in range(4):
                    px, py = corners[i]
                    qx, qy = corners[(i+1)%4]
                    crs = 0
                    # print("(%.3f, %.3f), (%.3f, %.3f)"%(px, py, qx, qy))
                    dx_last, dy_last = last_node.x - px, last_node.y - py
                    dx_now, dy_now = node.x - px, node.y - py
                    dx, dy = qx - px, qy - py
                    crs_last = dx_last * dy - dx * dy_last
                    crs_now = dx_now * dy - dx * dy_now
                    # print('(%.5f, %.5f) -> (%.5f, %.5f)'%(node.parent.x, node.parent.y, node.x, node.y), crs_last, crs_now)
                    if crs_last > 0 and crs_now < 0:
                        crs += 1
                    if crs_last < 0 and crs_now > 0:
                        crs += 1
                    
                    dx, dy = node.x - last_node.x, node.y - last_node.y
                    dx_p, dy_p = px - last_node.x, py - last_node.y
                    dx_q, dy_q = qx - last_node.x, qy - last_node.y
                    crs_p = dx_p * dy - dx * dy_p
                    crs_q = dx_q * dy - dx * dy_q
                    # print('(%.5f, %.5f) -> (%.5f, %.5f)'%(node.parent.x, node.parent.y, node.x, node.y), crs_p, crs_q)
                    if crs_p > 0 and crs_q < 0:
                        crs += 1
                    if crs_p < 0 and crs_q > 0:
                        crs += 1
                    if crs == 2:
                        return False
                    
        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta


def main(gx=6.0, gy=10.0):
    print("start " + __file__)

    # ====Search Path with RRT====
    obstacle_list = [
        (5, 5, 6, 6),
        (3, 6, 7, 8),
        (3, 8, 5, 9),
        (3, 5, 13, 6),
        (7, 5, 8, 10),
        (9, 5, 10, 6),
        (1, 6, 3, 12),
        (10, 7, 11, 14),
        (3, 10, 5, 14),
        (7, 2, 8, 6),
        (9, 3, 16, 4),
        (5, 12, 10, 14)
    ]  # [x,y,size(radius)]

    # Set Initial parameters
    rrt = RRT(
        start=[0, 0],
        goals=[[6.1, 10.1], [9,7]],
        rand_area=[[-2, 14], [0, 14]],
        obstacle_list=obstacle_list,
        expand_dis=0.5 ,
            goal_sample_rate=0,
                 max_iter=100000)
    # Set Initial parameters
    path = rrt.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

    show_path = True
    # Draw final path
    if show_path:
        rrt.draw_graph()
        if type(path)!= type(None):
            plt.plot([x for (x, y) in path], [y for (x, y) in path], 'r--')
        plt.grid(True)
        plt.savefig('/home/gaojiaxuan/onpolicy/onpolicy/scripts/gjx_tmp/rrt.png')


if __name__ == '__main__':
    main()

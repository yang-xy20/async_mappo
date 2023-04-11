from onpolicy.envs.gridworld.frontier.utils import generate_square
from onpolicy.envs.gridworld.frontier.utils import find_rectangle_obstacles
from onpolicy.utils.RRT.rrt import RRT
import numpy as np
from .utils import *
import random


class rrt_config(object):
    cluster_radius = 0
    rrt_max_iter = 10000
    rrt_num_targets = 500
    utility_edge_len = 7


def rrt_goal(map, unexplored, loc):
    H, W = map.shape
    map = map.astype(np.int32)

    obstacles = find_rectangle_obstacles(map)

    rrt = RRT(start=(loc[0] + 0.5, loc[1] + 0.5),
              goals=[],
              rand_area=((0, H), (0, W)),
              obstacle_list=obstacles,
              expand_dis=0.5,
              goal_sample_rate=-1,
              max_iter=rrt_config.rrt_max_iter)  # maybe more iterations?

    rrt_map = unexplored.copy().astype(np.int32)
    targets = rrt.select_frontiers(rrt_map, num_targets=rrt_config.rrt_num_targets)
    # print("targets: ", len(targets))

    clusters = get_frontier_cluster(targets, cluster_radius=rrt_config.cluster_radius)
    # print("clusters: ",len(clusters))

    if len(clusters) == 0:
        x, y = random.randint(0, H-1), random.randint(0, W-1)
        while map[x, y] == 1:
            x, y = random.randint(0, H-1), random.randint(0, W-1)
        goal = (x, y)
        return goal
    for cluster in clusters:
        center = cluster['center']
        # navigation cost
        nav_cost = l1distance(center, loc)
        # information gain
        mat = generate_square(H, W, center, rrt_config.utility_edge_len)
        area = mat.sum()
        info_gain = rrt_map[mat == 1].sum()
        info_gain /= area
        cluster['info_gain'] = info_gain
        cluster['nav_cost'] = nav_cost
    D = max([cluster['nav_cost'] for cluster in clusters])
    goal = None
    mx = -1e9
    for cluster in clusters:
        cluster['nav_cost'] /= D
        cluster['utility'] = cluster['info_gain'] - 1.0 * cluster['nav_cost']
        if mx < cluster['utility']:
            mx = cluster['utility']
            goal = cluster['center']

    return goal

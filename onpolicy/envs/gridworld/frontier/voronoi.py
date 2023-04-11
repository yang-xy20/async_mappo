from onpolicy.envs.gridworld.frontier.utils import generate_square
import numpy as np
from .utils import *
import random


def voronoi_goal(map, unexplored, locs, agent_id, steps, edge_len=7):
    num_agents = len(locs)

    dis = []
    vis = []
    for a, loc in enumerate(locs):
        _dis, _vis = bfs(map, loc, steps)
        dis.append(_dis)
        vis.append(_vis)
    
    H, W = map.shape
    my_grids = np.ones((H, W), dtype=np.int32)
    for a in range(num_agents):
        my_grids[dis[agent_id] > dis[a]] = 0

    utility = np.zeros((H, W), dtype=np.int32)
    frontiers = []
    for x in range(H):
        for y in range(W):
            if map[x, y] == 2 and vis[agent_id][x, y] and my_grids[x, y]:
                mat = generate_square(H, W, (x, y), edge_len)
                utility[x, y] = unexplored[mat == 1].sum()
                frontiers.append((x, y))
    mx = utility.max()
    value = [utility[x, y] for x, y in frontiers]
    candidates = [(x, y) for i, (x, y) in enumerate(frontiers) if value[i] == mx]
    if len(candidates) > 0:
        goal = random.choice(candidates)
    else:
        goal = random.randint(0, H-1), random.randint(0, W-1)
    return goal

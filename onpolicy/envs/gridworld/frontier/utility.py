from onpolicy.envs.gridworld.frontier.utils import generate_square
import numpy as np
from .utils import *
import random


def utility_goal(map, unexplored, loc, steps, edge_len=7):
    _, vis = bfs(map, loc, steps)
    H, W = map.shape
    utility = np.zeros((H, W), dtype=np.int32)
    frontiers = []
    for x in range(H):
        for y in range(W):
            if map[x, y] == 2 and vis[x, y]:
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

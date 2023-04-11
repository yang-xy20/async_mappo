import numpy as np
from .utils import *
import random


def nearest_goal(map, loc, steps):
    dis, vis = bfs(map, loc, steps)
    H, W = map.shape
    frontiers = []
    for x in range(H):
        for y in range(W):
            if map[x, y] == 2 and vis[x, y]:
                frontiers.append((x, y))
    if len(frontiers) == 0:
        goal = random.randint(0, H-1), random.randint(0, W-1)
        return goal
    dist = [dis[x, y] for x, y in frontiers]
    mi = min(dist)
    candidates = [(x, y) for i, (x, y) in enumerate(frontiers) if dist[i] == mi]
    goal = random.choice(candidates)
    return goal

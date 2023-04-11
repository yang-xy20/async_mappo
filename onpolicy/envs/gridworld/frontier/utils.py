import numpy as np
from queue import deque
import math


def l1distance(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])


def l2distance(x, y):
    return math.hypot(x[0]-y[0], x[1]-y[1])


def bfs(map, start, steps):
    sx, sy = start[0], start[1]
    assert map[sx, sy] != 1, ('start position should not be obstacle')
    que = deque([(sx, sy)])
    H, W = map.shape
    dis = np.ones((H, W), dtype=np.int32) * 1e9
    dis[sx, sy] = 0
    while len(que) > 0:
        x, y = que.popleft()
        neigh = [(x+dx, y+dy) for dx, dy in steps]
        neigh = [(x, y) for x, y in neigh if x >= 0 and x < H and y >
                 0 and y < W and map[x, y] != 1 and dis[x, y] == 1e9]
        for u, v in neigh:
            dis[u, v] = dis[x, y] + 1
            que.append((u, v))
    vis = (dis < 1e9)
    return dis, vis


def generate_square(H, W, loc, d):
    mat = np.zeros((H, W), dtype=np.int32)
    for x in range(H):
        for y in range(W):
            if max(abs(x-loc[0]), abs(y-loc[1])) <= d:
                mat[x, y] = 1
    return mat


def get_frontier_cluster(frontiers, cluster_radius=5.0):
    num_frontier = len(frontiers)
    clusters = []
    valid = [True for _ in range(num_frontier)]
    for i in range(num_frontier):
        if valid[i]:
            neigh = []
            for j in range(num_frontier):
                if valid[j] and l2distance(frontiers[i], frontiers[j]) <= cluster_radius:
                    valid[j] = False
                    neigh.append(frontiers[j])
            center = None
            min_r = 1e9
            for p in neigh:
                r = max([l2distance(p, q) for q in neigh])
                if r < min_r:
                    min_r = r
                    center = p
            if len(neigh) >= 5:
                clusters.append({'center': center, 'weight': len(neigh)})
    return clusters


def find_rectangle_obstacles(map):
    map = map.copy().astype(np.int32)
    map[map == 2] = 0
    H, W = map.shape
    obstacles = []
    covered = np.zeros((H, W), dtype=np.int32)
    pad = 0.01
    for x in range(H):
        for y in range(W):
            if map[x, y] == 1 and covered[x, y] == 0:
                x1 = x
                x2 = x
                while x2 < H-1 and map[x2 + 1, y] == 1:
                    x2 = x2 + 1
                y1 = y
                y2 = y
                while y2 < W-1 and map[x1: x2+1, y2 + 1].sum() == x2-x1+1:
                    y2 = y2 + 1
                covered[x1: x2 + 1, y1: y2 + 1] = 1
                obstacles.append((x1-pad, y1-pad, x2 + 1 + pad, y2 + 1 + pad))
    return obstacles

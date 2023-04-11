import numpy as np
from queue import deque
import pyastar2d

# class of APF(Artificial Potential Field)


class APF(object):
    def __init__(self, args):
        self.args = args

        self.cluster_radius = args.apf_cluster_radius
        self.k_attract = args.apf_k_attract
        self.k_agents = args.apf_k_agents
        self.AGENT_INFERENCE_RADIUS = args.apf_AGENT_INFERENCE_RADIUS
        self.num_iters = args.apf_num_iters
        self.repeat_penalty = args.apf_repeat_penalty
        self.dis_type = args.apf_dis_type

        self.num_agents = args.num_agents

    def distance(self, a, b):
        a = np.array(a)
        b = np.array(b)
        if self.dis_type == "l2":
            return np.sqrt(((a-b)**2).sum())
        elif self.dis_type == "l1":
            return abs(a-b).sum()

    def schedule(self, map, locations, steps, agent_id, penalty, full_path=True):
        '''
        APF to schedule path for agent agent_id
        map: H x W
            - 0 for explored & available cell
            - 1 for obstacle
            - 2 for target (frontier)
        locations: num_agents x 2
        steps: available actions
        penalty: repeat penalty
        full_path: default True, False for single step (i.e., next cell)
        '''
        H, W = map.shape

        # find available targets
        vis = np.zeros((H, W), dtype=np.uint8)
        que = deque([])
        x, y = locations[agent_id]
        vis[x, y] = 1
        que.append((x, y))
        while len(que) > 0:
            x, y = que.popleft()
            for dx, dy in steps:
                x1 = x + dx
                y1 = y + dy
                if vis[x1, y1] == 0 and map[x1, y1] in [0, 2]:
                    vis[x1, y1] = 1
                    que.append((x1, y1))

        targets = []
        for i in range(H):
            for j in range(W):
                if map[i, j] == 2 and vis[i, j] == 1:
                    targets.append((i, j))
        # clustering
        clusters = []
        num_targets = len(targets)
        valid = [True for _ in range(num_targets)]
        for i in range(num_targets):
            if valid[i]:
                # not clustered
                chosen_targets = []
                for j in range(num_targets):
                    if valid[j] and self.distance(targets[i], targets[j]) <= self.cluster_radius:
                        valid[j] = False
                        chosen_targets.append(targets[j])
                min_r = 1e6
                center = None
                for a in chosen_targets:
                    max_d = max([self.distance(a, b) for b in chosen_targets])
                    if max_d < min_r:
                        min_r = max_d
                        center = a
                clusters.append({"center": center, "weight": len(chosen_targets)})

        # potential
        num_clusters = len(clusters)
        potential = np.zeros((H, W))
        potential[map == 1] = 1e6

        # potential of targets & obstacles (wave-front dist)
        for cluster in clusters:
            sx, sy = cluster["center"]
            w = cluster["weight"]
            dis = np.ones((H, W), dtype=np.int64) * 1e6
            dis[sx, sy] = 0
            que = deque([(sx, sy)])
            while len(que) > 0:
                (x, y) = que.popleft()
                for dx, dy in steps:
                    x1 = x + dx
                    y1 = y + dy
                    if dis[x1, y1] == 1e6 and map[x1, y1] in [0, 2]:
                        dis[x1, y1] = dis[x, y]+1
                        que.append((x1, y1))
            dis[sx, sy] = 1e6
            dis = 1 / dis
            dis[sx, sy] = 0
            potential[map != 1] -= dis[map != 1] * self.k_attract * w

        # potential of agents
        for x in range(H):
            for y in range(W):
                for agent_loc in locations:
                    d = self.distance(agent_loc, (x, y))
                    if d <= self.AGENT_INFERENCE_RADIUS:
                        potential[x, y] += self.k_agents * (self.AGENT_INFERENCE_RADIUS - d)

        # repeat penalty
        potential += penalty

        # schedule path
        it = 1
        current_loc = locations[agent_id]
        current_potential = 1e4
        minDis2Target = 1e6
        path = [(current_loc[0], current_loc[1])]
        while it <= self.num_iters and minDis2Target > 1:
            it = it + 1
            potential[current_loc[0], current_loc[1]] += self.repeat_penalty
            best_neigh = None
            min_potential = 1e6
            for dx, dy in steps:
                neighbor_loc = (current_loc[0] + dx, current_loc[1] + dy)
                if map[neighbor_loc[0], neighbor_loc[1]] == 1:
                    continue
                if min_potential > potential[neighbor_loc[0], neighbor_loc[1]]:
                    min_potential = potential[neighbor_loc[0], neighbor_loc[1]]
                    best_neigh = neighbor_loc
            if current_potential > min_potential:
                current_potential = min_potential
                current_loc = best_neigh
                path.append(best_neigh)
            for tar in targets:
                l = self.distance(current_loc, tar)
                if l == 0:
                    continue
                minDis2Target = min(minDis2Target, l)
                if l <= 1:
                    path.append((tar[0], tar[1]))
                    break
            if not full_path and len(path) > 1:
                return path[1]  # next grid
        random_plan = False
        if minDis2Target > 1:
            random_plan = True
        for i in range(agent_id):
            if locations[i][0] == locations[agent_id][0] and locations[i][1] == locations[agent_id][1]:
                random_plan = True  # two agents are at the same location, replan
        if random_plan:
            # if not reaching a frontier, randomly pick a traget as goal
            if num_targets == 0:
                targets = [(np.random.randint(0, H), np.random.randint(0, W))]
                num_targets = 1
            w = np.random.randint(0, num_targets)
            goal = targets[w]
            temp_map = np.ones((H, W), dtype=np.float32)
            temp_map[temp_map == 1] = 1000000
            temp_map[temp_map == 2] = 1.0
            path = pyastar2d.astar_path(temp_map, locations[agent_id], goal, allow_diagonal=False)
            if len(path) == 1:
                path = (locations[agent_id], goal)
            return path
        return path

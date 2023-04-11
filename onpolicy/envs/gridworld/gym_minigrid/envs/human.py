from onpolicy.envs.gridworld.gym_minigrid.minigrid import *
from icecream import ic
import collections
import math
from copy import deepcopy

class HumanEnv(MiniGridEnv):
    """
    Environment in which the agent is instructed to go to a given object
    named using an English text string
    """

    def __init__(
        self,
        num_agents=2,
        num_preies=2,
        num_obstacles=4,
        direction_alpha=0.5,
        use_direction_reward = False,
        use_human_command=False,
        coverage_discounter=0.1,
        size=19
    ):
        self.size = size
        self.num_preies = num_preies
        self.use_direction_reward = use_direction_reward
        self.direction_alpha = direction_alpha
        self.use_human_command = use_human_command
        self.coverage_discounter = coverage_discounter
        # initial the covering rate
        self.covering_rate = 0
        # Reduce obstacles if there are too many
        if num_obstacles <= size/2 + 1:
            self.num_obstacles = int(num_obstacles)
        else:
            self.num_obstacles = int(size/2)

        super().__init__(
            num_agents=num_agents,
            grid_size=size,
            max_steps=math.floor(((size-2)**2) / num_agents * 2),
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)

        # initial the cover_grid
        self.cover_grid = np.zeros([width,height])
        for j in range(0, height):
            for i in range(0, width):
                if self.grid.get(i,j) != None and self.grid.get(i,j).type == 'wall':
                    self.cover_grid[j,i] = 1.0
        self.cover_grid_initial = self.cover_grid.copy()
        self.num_none = collections.Counter(self.cover_grid_initial.flatten())[0.]
        # import pdb; pdb.set_trace()

        # Types and colors of objects we can generate
        types = ['key']

        objs = []
        objPos = []

        # Until we have generated all the objects
        while len(objs) < self.num_preies:
            objType = self._rand_elem(types)
            objColor = self._rand_elem(COLOR_NAMES)

            # If this object already exists, try again
            if (objType, objColor) in objs:
                continue

            if objType == 'key':
                obj = Key(objColor)
            elif objType == 'box':
                obj = Box(objColor)
            elif objType == 'ball':
                obj = Ball(objColor)

            pos = self.place_obj(obj)
            objs.append((objType, objColor))
            objPos.append(pos)

        # Place obstacles
        self.obstacles = []
        for i_obst in range(self.num_obstacles):
            self.obstacles.append(Obstacle())
            pos = self.place_obj(self.obstacles[i_obst], max_tries=100)

        self.occupy_grid = self.grid.copy()
        # Randomize the agent start position and orientation
        self.place_agent()

        # Choose a random object to be picked up
        objIdx = self._rand_int(0, len(objs))
        self.targetType, self.target_color = objs[objIdx]
        self.target_pos = objPos[objIdx]
        
        # direction
        array_direction = np.array([[0,1], [0,-1], [1,0], [-1,0], [1,1], [1,-1], [-1,1], [-1,-1]])
        self.direction = []
        self.direction_encoder = []
        self.direction_index = []
        for agent_id in range(self.num_agents):
            center_pos = np.array([int((self.size-1)/2),int((self.size-1)/2)])
            direction = np.sign(center_pos - self.agent_pos[agent_id])
            direction_index = np.argmax(np.all(np.where(array_direction == direction, True, False), axis=1))
            direction_encoder = np.eye(8)[direction_index]
            self.direction_index.append(direction_index)
            self.direction.append(direction)
            self.direction_encoder.append(direction_encoder)

        # text
        descStr = '%s %s' % (self.target_color, self.targetType)
        self.mission = 'go to the %s' % descStr
        # print(self.mission)

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        
        rewards = []

        for agent_id in range(self.num_agents):
            ax, ay = self.agent_pos[agent_id]
            tx, ty = self.target_pos
            if self.cover_grid[ay,ax] == 0:
                reward += self.coverage_discounter
                self.cover_grid[ay, ax] = 1.0
                self.covering_rate = collections.Counter((self.cover_grid - self.cover_grid_initial).flatten())[1] / self.num_none

            # if abs(ax - tx) < 1 and abs(ay - ty) < 1:
            #     reward += 1.0 
            #     self.num_reach_goal += 1
            #     # done = True
                    
            rewards.append(reward)

        rewards = [[np.sum(rewards)]] * self.num_agents

        dones = [done for agent_id in range(self.num_agents)]
        
        info['num_reach_goal'] = self.num_reach_goal
        info['covering_rate'] = self.covering_rate
        info['num_same_direction'] = self.num_same_direction

        return obs, rewards, dones, info




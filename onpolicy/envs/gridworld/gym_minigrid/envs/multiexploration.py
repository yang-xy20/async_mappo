#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import time
import pyastar2d
from onpolicy.envs.gridworld.gym_minigrid.minigrid import *
from .multiroom import *
from icecream import ic
import cv2
import random
import copy
import numpy as np

from onpolicy.envs.gridworld.frontier.apf import APF
from onpolicy.envs.gridworld.frontier.utility import utility_goal
from onpolicy.envs.gridworld.frontier.rrt import rrt_goal
from onpolicy.envs.gridworld.frontier.nearest import nearest_goal
from onpolicy.envs.gridworld.frontier.voronoi import voronoi_goal


class MultiExplorationEnv(MultiRoomEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(
        self,
        grid_size,
        max_steps,
        local_step_num,
        agent_view_size,
        num_obstacles,
        num_agents=2,
        agent_pos=None,
        goal_pos=None,
        use_merge=True,
        use_merge_plan=True,
        use_constrict_map=True,
        use_fc_net=False,
        use_agent_id=False,
        use_stack=False,
        use_orientation=False,
        use_same_location=True,
        use_complete_reward=True,
        use_multiroom=False,
        use_irregular_room=False,
        use_time_penalty=False,
        use_overlap_penalty=False,
        use_agent_obstacle = False,
        astar_cost_mode='normal'
    ):
        self.grid_size = grid_size
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        self.door_size = 1
        self.max_steps = max_steps
        self.use_same_location = use_same_location
        self.use_complete_reward = use_complete_reward
        self.use_multiroom = use_multiroom
        self.use_irregular_room = use_irregular_room
        self.use_time_penalty = use_time_penalty
        self.use_overlap_penalty = use_overlap_penalty
        self.use_merge = use_merge
        self.use_merge_plan = use_merge_plan
        self.use_constrict_map = use_constrict_map
        self.astar_cost_mode = astar_cost_mode
        self.use_agent_obstacle = use_agent_obstacle
        self.astar_utility_radius = agent_view_size // 2 + 1
        self.maxNum = 5
        self.minNum = 2

        if num_obstacles <= grid_size/2 + 1:
            self.num_obstacles = int(num_obstacles)
        else:
            self.num_obstacles = int(grid_size/2)

        super().__init__(minNumRooms=4,
                         maxNumRooms=7,
                         maxRoomSize=8,
                         grid_size=grid_size,
                         max_steps=max_steps,
                         num_agents=num_agents,
                         agent_view_size=agent_view_size,
                         use_merge=use_merge,
                         use_merge_plan=use_merge_plan,
                         use_constrict_map=use_constrict_map,
                         use_fc_net=use_fc_net,
                         use_agent_id=use_agent_id,
                         use_stack=use_stack,
                         use_orientation=use_orientation,

                         )
        if use_agent_id:
            self.augment = [255 / (np.array([agent_id+1 for agent_id in range(self.num_agents)]).sum()) for _ in range(self.num_agents)]
        else:
            self.augment = [255 / (agent_id+1)  for agent_id in range(self.num_agents)]
        self.target_ratio = 0.90
        self.merge_ratio = 0
        self.merge_reward = 0
        self.episode_no = 0
        self.agent_ratio = np.zeros((num_agents))
        self.agent_reward = np.zeros((num_agents))
        self.agent_ratio_step = np.ones((num_agents)) * max_steps
        self.merge_ratio_step = max_steps
        self.merge_ratio_up_step = max_steps
        self.max_steps = max_steps
        self.local_step_num = local_step_num
        self.use_agent_id = use_agent_id
    def overall_gen_grid(self, width, height):

        if self.use_multiroom:
            self.multiroom_gen_grid(width, height)
        elif self.use_irregular_room:
            self.irregular_room_gen_grid(width, height)
        else:
            # Create the grid
            self.grid = Grid(width, height)
            
            # Generate the surrounding walls
            self.grid.horz_wall(0, 0)
            self.grid.horz_wall(0, height - 1)
            self.grid.vert_wall(0, 0)
            self.grid.vert_wall(width - 1, 0)

            w = self._rand_int(self.minNum, self.maxNum)
            h = self._rand_int(self.minNum, self.maxNum)

            room_w = width // w
            room_h = height // h

            # For each row of rooms
            for j in range(0, h):

                # For each column
                for i in range(0, w):
                    xL = i * room_w
                    yT = j * room_h
                    xR = xL + room_w
                    yB = yT + room_h
                    # if self.scene_id = 1:
                    # Bottom wall and door
                    if i + 1 < w:

                        self.grid.vert_wall(xR, yT, room_h)
                        pos = (xR, self._rand_int(yT + 1, yB))

                        for s in range(self.door_size):
                            self.grid.set(*pos, None)
                            pos = (pos[0], pos[1] + 1)

                    # Bottom wall and door
                    if j + 1 < h:

                        self.grid.horz_wall(xL, yB, room_w)
                        pos = (self._rand_int(xL + 1, xR), yB)
                        self.grid.set(*pos, None)

            # Randomize the player start position and orientation
            if self._agent_default_pos is not None:
                self.agent_pos = self._agent_default_pos
                for i in range(self.num_agents):
                    self.grid.set(*self._agent_default_pos[i], None)
                self.agent_dir = [self._rand_int(0, 4) for i in range(
                    self.num_agents)]  # assuming random start direction
            else:
                self.place_agent(use_same_location=self.use_same_location)

            # place object
            self.obstacles = []
            for i_obst in range(self.num_obstacles):
                self.obstacles.append(Obstacle())
                pos = self.place_obj(self.obstacles[i_obst], max_tries=100)

            self.mission = 'Reach the goal'

    def reset(self):
        self.explorable_size = 0
        obs = MiniGridEnv.reset(self, choose=True)

        self.num_step = 0
        self.get_ratio = 0
        self.episode_no += 1
        self.target_ratio = 0.90
        self.target_up_ratio = 0.98
        merge_overlap = False
        self.prev_overlap_area = 0
        #self.gt_map = self.grid.encode()[:,:,0].T
        self.agent_local_map = np.zeros(
            (self.num_agents, self.agent_view_size, self.agent_view_size, 3))
        #self.pad_gt_map = np.pad(self.gt_map,((self.agent_view_size, self.agent_view_size), (self.agent_view_size,self.agent_view_size)) , constant_values=(0,0))

        # init local map
        self.explored_each_map = []
        self.obstacle_each_map = []
        self.previous_explored_each_map = []
        current_agent_pos = []
        self.overlap_delta_each_map = []
        # APF repeat penalty.
        self.ft_goals = [None for _ in range(self.num_agents)]
        self.apf_penalty = np.zeros((
            self.num_agents,
            self.width + 2*self.agent_view_size,
            self.height + 2*self.agent_view_size
        ))
        self.agent_plan_explored = np.zeros((self.num_agents, self.width,self.height))
        self.agent_plan_obstacle = np.zeros((self.num_agents, self.width,self.height))
        self.ft_agent_plan_explored = np.zeros((self.num_agents, self.width+ 2*self.agent_view_size,self.height+ 2*self.agent_view_size))
        self.ft_agent_plan_obstacle = np.zeros((self.num_agents, self.width+ 2*self.agent_view_size,self.height+ 2*self.agent_view_size))
        for i in range(self.num_agents):
            self.explored_each_map.append(
                np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))
            self.obstacle_each_map.append(
                np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))
            self.previous_explored_each_map.append(
                np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))
            self.overlap_delta_each_map.append(
                np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))

        for i in range(self.num_agents):
            local_map = np.rot90(obs[i]['image'][:, :, 0].T, 3)
            pos = [self.agent_pos[i][1] + self.agent_view_size,
                   self.agent_pos[i][0] + self.agent_view_size]
            direction = self.agent_dir[i]
            current_agent_pos.append(pos)
            # adjust angle
            local_map = np.rot90(local_map, 4-direction)
            if direction == 0:  # Facing right
                for x in range(self.agent_view_size):
                    for y in range(self.agent_view_size):
                        if local_map[x][y] == 0:
                            continue
                        else:
                            self.explored_each_map[i][x+pos[0] -
                                                    self.agent_view_size//2][y+pos[1]] = (i+1)*self.augment[i]
                            self.overlap_delta_each_map[i][x+pos[0] -
                                                      self.agent_view_size//2][y+pos[1]] = 1
                            if local_map[x][y] != 20:
                                self.obstacle_each_map[i][x+pos[0] -
                                                        self.agent_view_size//2][y+pos[1]] = (i+1)*self.augment[i]
            if direction == 1:  # Facing down
                for x in range(self.agent_view_size):
                    for y in range(self.agent_view_size):
                        if local_map[x][y] == 0:
                            continue
                        else:
                            self.explored_each_map[i][x+pos[0]][y+pos[1] -
                                                                self.agent_view_size//2] = (i+1)*self.augment[i]
                            self.overlap_delta_each_map[i][x+pos[0]][y+pos[1] -
                                                                self.agent_view_size//2] = 1 
                            if local_map[x][y] != 20:
                                self.obstacle_each_map[i][x+pos[0]][y+pos[1] -
                                                                    self.agent_view_size//2] = (i+1)*self.augment[i]
            if direction == 2:  # Facing left
                for x in range(self.agent_view_size):
                    for y in range(self.agent_view_size):
                        if local_map[x][y] == 0:
                            continue
                        else:
                            self.explored_each_map[i][x+pos[0]-self.agent_view_size //
                                                    2][y+pos[1]-self.agent_view_size+1] = (i+1)*self.augment[i]
                            self.overlap_delta_each_map[i][x+pos[0]-self.agent_view_size //
                                                      2][y+pos[1]-self.agent_view_size+1] = 1
                            if local_map[x][y] != 20:
                                self.obstacle_each_map[i][x+pos[0]-self.agent_view_size //
                                                        2][y+pos[1]-self.agent_view_size+1] = (i+1)*self.augment[i]
            if direction == 3:  # Facing up
                for x in range(self.agent_view_size):
                    for y in range(self.agent_view_size):
                        if local_map[x][y] == 0:
                            continue
                        else:
                            self.explored_each_map[i][x+pos[0]-self.agent_view_size +
                                                    1][y+pos[1]-self.agent_view_size//2] = (i+1)*self.augment[i]
                            self.overlap_delta_each_map[i][x+pos[0]-self.agent_view_size +
                                                      1][y+pos[1]-self.agent_view_size//2] = 1
                            if local_map[x][y] != 20:
                                self.obstacle_each_map[i][x+pos[0]-self.agent_view_size +
                                                        1][y+pos[1]-self.agent_view_size//2] = (i+1)*self.augment[i]
                                
            for j in range(3):
                mmap = np.rot90(obs[i]['image'][:, :, j].T, 3)
                mmap = np.rot90(mmap, 4-direction)
                self.agent_local_map[i, :, :, j] = mmap
        explored_all_map = np.zeros((self.width + 2*self.agent_view_size,
                                    self.height + 2*self.agent_view_size))
        obstacle_all_map = np.zeros((self.width + 2*self.agent_view_size,
                                    self.height + 2*self.agent_view_size))
        self.previous_all_map = np.zeros(
            (self.width + 2*self.agent_view_size, self.width + 2*self.agent_view_size))

        # APF penalty
        for i in range(self.num_agents):
            x, y = current_agent_pos[i]
            self.apf_penalty[i, x, y] = 5.0  # constant

        for i in range(self.num_agents):
            if self.use_agent_id:
                explored_all_map += self.explored_each_map[i]
                obstacle_all_map += self.obstacle_each_map[i]
            else:
                explored_all_map = np.maximum( explored_all_map, self.explored_each_map[i])
                obstacle_all_map = np.maximum( obstacle_all_map, self.obstacle_each_map[i])
        self.explored_map = np.array(explored_all_map).astype(int)[
            self.agent_view_size: self.width+self.agent_view_size, self.agent_view_size: self.width+self.agent_view_size]
        
        if self.episode_no > 1 :
            if 'merge_overlap_ratio' in self.info.keys():
                merge_overlap_ratio = self.info['merge_overlap_ratio']
                merge_overlap = True

        self.info = {}
        self.info['explored_all_map'] = np.array(explored_all_map)
        self.info['current_agent_pos'] = np.array(current_agent_pos)
        self.info['explored_each_map'] = np.array(self.explored_each_map)
        self.info['obstacle_all_map'] = np.array(obstacle_all_map)
        self.info['obstacle_each_map'] = np.array(self.obstacle_each_map)
        self.info['agent_direction'] = np.array(self.agent_dir)
        self.info['agent_local_map'] = self.agent_local_map

        self.info['merge_explored_ratio'] = self.merge_ratio
        self.info['merge_explored_reward'] = self.merge_reward
        self.info['agent_explored_ratio'] = self.agent_ratio
        self.info['agent_explored_reward'] = self.agent_reward
        if merge_overlap:
            self.info['merge_overlap_ratio'] = merge_overlap_ratio
        
        self.merge_ratio = 0
        self.merge_reward = 0
        self.agent_ratio = np.zeros((self.num_agents))
        self.agent_reward = np.zeros((self.num_agents))
        self.agent_ratio_step = np.ones((self.num_agents)) * self.max_steps
        self.merge_ratio_step = self.max_steps
        self.merge_ratio_up_step = self.max_steps
        self.ft_info = copy.deepcopy(self.info)
        return obs, self.info

    def step(self, action):
        obs, reward, done, self.info = MiniGridEnv.step(self, action)
        self.explored_each_map_t = []
        self.obstacle_each_map_t = []
        current_agent_pos = []
        each_agent_rewards = []
        self.num_step += 1
        reward_obstacle_each_map = np.zeros(
            (self.num_agents, self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        delta_reward_each_map = np.zeros(
            (self.num_agents, self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        reward_explored_each_map = np.zeros(
            (self.num_agents, self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size))
        explored_all_map = np.zeros((self.width + 2*self.agent_view_size,
                                    self.height + 2*self.agent_view_size))
        obstacle_all_map = np.zeros((self.width + 2*self.agent_view_size,
                                    self.height + 2*self.agent_view_size))

        for i in range(self.num_agents):
            self.explored_each_map_t.append(
                np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))
            self.obstacle_each_map_t.append(
                np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)))
        for i in range(self.num_agents):
            local_map = np.rot90(obs[i]['image'][:, :, 0].T, 3)

            pos = [self.agent_pos[i][1] + self.agent_view_size,
                   self.agent_pos[i][0] + self.agent_view_size]
            current_agent_pos.append(pos)
            direction = self.agent_dir[i]
            # adjust angle
            local_map = np.rot90(local_map, 4-direction)
            if direction == 0:  # Facing right
                for x in range(self.agent_view_size):
                    for y in range(self.agent_view_size):
                        if local_map[x][y] == 0:
                            continue
                        else:
                            self.explored_each_map_t[i][x+pos[0] -
                                                        self.agent_view_size//2][y+pos[1]] = (i+1)*self.augment[i]
                            self.overlap_delta_each_map[i][x+pos[0] -
                                                        self.agent_view_size//2][y+pos[1]] = 1
                            if local_map[x][y] != 20:
                                self.obstacle_each_map_t[i][x+pos[0] -
                                                            self.agent_view_size//2][y+pos[1]] = (i+1)*self.augment[i]
            if direction == 1:  # Facing down
                for x in range(self.agent_view_size):
                    for y in range(self.agent_view_size):
                        if local_map[x][y] == 0:
                            continue
                        else:
                            self.explored_each_map_t[i][x+pos[0]][y+pos[1] -
                                                                  self.agent_view_size//2] = (i+1)*self.augment[i]
                            self.overlap_delta_each_map[i][x+pos[0]][y+pos[1] -
                                                                  self.agent_view_size//2] = 1
                            if local_map[x][y] != 20:
                                self.obstacle_each_map_t[i][x+pos[0]][y+pos[1] -
                                                                      self.agent_view_size//2] = (i+1)*self.augment[i]
            if direction == 2:  # Facing left
                for x in range(self.agent_view_size):
                    for y in range(self.agent_view_size):
                        if local_map[x][y] == 0:
                            continue
                        else:
                            self.explored_each_map_t[i][x+pos[0]-self.agent_view_size //
                                                        2][y+pos[1]-self.agent_view_size+1] = (i+1)*self.augment[i]
                            self.overlap_delta_each_map[i][x+pos[0]-self.agent_view_size //
                                                        2][y+pos[1]-self.agent_view_size+1] = 1
                            if local_map[x][y] != 20:
                                self.obstacle_each_map_t[i][x+pos[0]-self.agent_view_size //
                                                            2][y+pos[1]-self.agent_view_size+1] = (i+1)*self.augment[i]
            if direction == 3:  # Facing up
                for x in range(self.agent_view_size):
                    for y in range(self.agent_view_size):
                        if local_map[x][y] == 0:
                            continue
                        else:

                            self.explored_each_map_t[i][x+pos[0]-self.agent_view_size +
                                                        1][y+pos[1]-self.agent_view_size//2] = (i+1)*self.augment[i]
                            self.overlap_delta_each_map[i][x+pos[0]-self.agent_view_size +
                                                        1][y+pos[1]-self.agent_view_size//2] = 1
                            if local_map[x][y] != 20:
                                self.obstacle_each_map_t[i][x+pos[0]-self.agent_view_size +
                                                            1][y+pos[1]-self.agent_view_size//2] = (i+1)*self.augment[i]
            for j in range(3):
                mmap = np.rot90(obs[i]['image'][:, :, j].T, 3)
                mmap = np.rot90(mmap, 4 - direction)
                self.agent_local_map[i, :, :, j] = mmap

        for i in range(self.num_agents):
            self.explored_each_map[i] = np.maximum(
                self.explored_each_map[i], self.explored_each_map_t[i])
            self.obstacle_each_map[i] = np.maximum(
                self.obstacle_each_map[i], self.obstacle_each_map_t[i])

            reward_explored_each_map[i] = self.explored_each_map[i].copy()
            reward_explored_each_map[i][reward_explored_each_map[i] != 0] = 1

            reward_previous_explored_each_map = self.previous_explored_each_map[i].copy()
            reward_previous_explored_each_map[reward_previous_explored_each_map != 0] = 1

            reward_obstacle_each_map[i] = self.obstacle_each_map[i].copy()
            reward_obstacle_each_map[i][reward_obstacle_each_map[i] != 0] = 1

            delta_reward_each_map[i] = reward_explored_each_map[i] - reward_obstacle_each_map[i]

            each_agent_rewards.append(
                (np.array(delta_reward_each_map[i]) - np.array(reward_previous_explored_each_map)).sum())
            self.previous_explored_each_map[i] = self.explored_each_map[i] - \
                self.obstacle_each_map[i]

        for i in range(self.num_agents):
            if self.use_agent_id:
                explored_all_map += self.explored_each_map[i]
                obstacle_all_map += self.obstacle_each_map[i]
            else:
                explored_all_map = np.maximum(explored_all_map, self.explored_each_map[i])
                obstacle_all_map = np.maximum(obstacle_all_map, self.obstacle_each_map[i])

        reward_explored_all_map = explored_all_map.copy()
        reward_explored_all_map[reward_explored_all_map != 0] = 1

        reward_obstacle_all_map = obstacle_all_map.copy()
        reward_obstacle_all_map[reward_obstacle_all_map != 0] = 1

        delta_reward_all_map = reward_explored_all_map - reward_obstacle_all_map

        reward_previous_all_map = self.previous_all_map.copy()
        reward_previous_all_map[reward_previous_all_map != 0] = 1

        merge_explored_reward = (np.array(delta_reward_all_map) -
                                 np.array(reward_previous_all_map)).sum()
        self.previous_all_map = explored_all_map - obstacle_all_map
        self.explored_map = np.array(explored_all_map).astype(int)[
            self.agent_view_size: self.width + self.agent_view_size, self.agent_view_size: self.width + self.agent_view_size]

        self.info = {}
        self.info['explored_all_map'] = np.array(explored_all_map)
        self.info['current_agent_pos'] = np.array(current_agent_pos)
        self.info['explored_each_map'] = np.array(self.explored_each_map)
        self.info['obstacle_all_map'] = np.array(obstacle_all_map)
        self.info['obstacle_each_map'] = np.array(self.obstacle_each_map)
        self.info['agent_direction'] = np.array(self.agent_dir)
        self.info['agent_local_map'] = self.agent_local_map
        if self.use_time_penalty:
            self.info['agent_explored_reward'] = np.array(each_agent_rewards) * 0.02 - 0.01
            self.info['merge_explored_reward'] = merge_explored_reward * 0.02 - 0.01
        else:
            self.info['agent_explored_reward'] = np.array(each_agent_rewards) * 0.02
            self.info['merge_explored_reward'] = merge_explored_reward * 0.02

        if self.use_irregular_room:
            self.no_wall_size = self.explorable_size

        if (delta_reward_all_map.sum() / self.no_wall_size) >= self.target_ratio:
            if self.merge_ratio_step == self.max_steps:
                self.merge_ratio_step = self.num_step
                self.info['merge_explored_ratio_step'] = self.merge_ratio_step
                
                overlap_delta_map = np.sum(delta_reward_each_map, axis=0)
                self.info['merge_overlap_ratio'] = (overlap_delta_map > 1).sum() / delta_reward_all_map.sum()
                
                # if self.use_complete_reward:
                #     self.info['merge_explored_reward'] += 0.1 * \
                #         (delta_reward_all_map.sum() / self.no_wall_size)
                # if self.num_step % self.local_step_num == 0:
                #     done = True
        elif self.use_overlap_penalty and self.num_step % self.local_step_num==0:
                average_overlap_area = []
                for a in range(self.num_agents):
                    for b in range(a+1,self.num_agents):
                        overlap_delta_map = self.overlap_delta_each_map[a]+self.overlap_delta_each_map[b]                
                        overlap_area = (overlap_delta_map > 1).sum()
                        average_overlap_area.append(overlap_area)
                self.info['merge_explored_reward'] -= 0.001 * (np.array(average_overlap_area).mean())
                self.overlap_delta_each_map = [np.zeros((self.width + 2*self.agent_view_size, self.height + 2*self.agent_view_size)) for _ in range(self.num_agents)]

        if (delta_reward_all_map.sum() / self.no_wall_size) >= self.target_up_ratio:
            if self.merge_ratio_up_step == self.max_steps:
                self.merge_ratio_up_step = self.num_step
                self.info['merge_explored_ratio_step_0.98'] = self.merge_ratio_up_step
                if self.use_complete_reward:
                    self.info['merge_explored_reward'] += 0.1 * \
                        (delta_reward_all_map.sum() / self.no_wall_size)
            
        for i in range(self.num_agents):
            # (self.width * self.height)
            self.agent_ratio = delta_reward_each_map[i].sum() / self.no_wall_size
            if (delta_reward_each_map[i].sum() / self.no_wall_size) >= self.target_ratio:
                if self.agent_ratio_step[i] == self.max_steps:
                    self.agent_ratio_step[i] = self.num_step
                    self.info["agent{}_explored_ratio_step".format(i)] = self.agent_ratio_step[i]
                # if self.use_complete_reward:
                #     self.info['agent_explored_reward'][i] += 0.1 * (reward_explored_each_map[i].sum() / (self.width * self.height))

        self.agent_reward = self.info['agent_explored_reward']
        self.merge_reward = self.info['merge_explored_reward']
        self.merge_ratio = delta_reward_all_map.sum() / self.no_wall_size  # (self.width * self.height)
      
        self.info['merge_explored_ratio'] = self.merge_ratio
        self.info['agent_explored_ratio'] = self.agent_ratio
        
        self.ft_info = copy.deepcopy(self.info)
        if self.num_step >= self.max_steps:
            done = True
        return obs, reward, done, self.info

    def ft_get_short_term_goals(self, args, mode=""):
        '''
        frontier-based methods compute actions
        '''
        self.info = self.ft_info
        self.use_merge_plan=args.use_merge_plan
        replan = [False for _ in range(self.num_agents)]
        current_agent_pos = self.info["current_agent_pos"]
        goals = [None for _ in range(self.num_agents)]
        for agent_id in range(self.num_agents):
            if self.use_constrict_map:
                self.ft_agent_plan_explored[agent_id] = np.maximum(self.ft_agent_plan_explored[agent_id],(self.info['explored_each_map'][agent_id] > 0).astype(np.int32))
                self.ft_agent_plan_obstacle[agent_id] = np.maximum(self.ft_agent_plan_obstacle[agent_id],(self.info['obstacle_each_map'][agent_id] > 0).astype(np.int32))
                for b in range(agent_id+1,self.num_agents):
                    rel_dis = self.distance(current_agent_pos[agent_id] ,current_agent_pos[b])
                    if rel_dis < 8:
                        self.ft_agent_plan_explored[agent_id] = np.maximum(self.ft_agent_plan_explored[agent_id],(self.info['explored_each_map'][b] > 0).astype(np.int32))
                        self.ft_agent_plan_obstacle[agent_id] = np.maximum(self.ft_agent_plan_obstacle[agent_id],(self.info['obstacle_each_map'][b] > 0).astype(np.int32))
                explored = self.ft_agent_plan_explored[agent_id]
                obstacle = self.ft_agent_plan_obstacle[agent_id]

            elif self.use_merge_plan:
                explored = (self.info['explored_all_map'] > 0).astype(np.int32)
                obstacle = (self.info['obstacle_all_map'] > 0).astype(np.int32)
            else:
                explored = (self.info['explored_each_map'][agent_id] > 0).astype(np.int32)
                obstacle = (self.info['obstacle_each_map'][agent_id] > 0).astype(np.int32)
            if self.use_agent_obstacle:
                for a in range(self.num_agents):
                    if a != agent_id:
                        obstacle[current_agent_pos[a][0], current_agent_pos[a][1]] = 1

            H, W = explored.shape
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            map = np.ones((H, W)).astype(np.int32) * 3  # 3 for unknown area
            map[explored == 1] = 0  # 0 for explored area
            map[obstacle == 1] = 1  # 1 for obstacles
            # Set frontiers.
            for x in range(H):
                for y in range(W):
                    if map[x, y] == 0:
                        neighbors = [(x+dx, y+dy) for dx, dy in steps]
                        if sum([(map[u, v] == 3) for u, v in neighbors]) > 0:
                            map[x, y] = 2  # 2 for targets (frontiers)
            map[:self.agent_view_size, :] = 1
            map[H-self.agent_view_size:, :] = 1
            map[:, :self.agent_view_size] = 1
            map[:, W-self.agent_view_size:] = 1
            unexplored = (map == 3).astype(np.int32)
            map[map == 3] = 0  # set unknown area to explorable

            if self.num_step >= 1:
                if (map[self.ft_goals[agent_id][0], self.ft_goals[agent_id][1]] != 2) and\
                (unexplored[self.ft_goals[agent_id][0], self.ft_goals[agent_id][1]] == 0):
                    replan[agent_id] = True

            if replan[agent_id] or self.ft_goals[agent_id] is None:
                if mode == 'apf':
                    apf = APF(args)
                    path = apf.schedule(map, current_agent_pos, steps,
                                        agent_id, self.apf_penalty[agent_id])
                    goal = path[-1]
                elif mode == 'utility':
                    goal = utility_goal(map, unexplored, current_agent_pos[agent_id], steps)
                elif mode == 'nearest':
                    goal = nearest_goal(map, current_agent_pos[agent_id], steps)
                elif mode == 'rrt':
                    goal = rrt_goal(map, unexplored, current_agent_pos[agent_id])
                elif mode == 'voronoi':
                    goal = voronoi_goal(map, unexplored, current_agent_pos, agent_id, steps)
                goals[agent_id] = goal
            else:
                goals[agent_id] = self.ft_goals[agent_id]
        self.ft_goals = goals.copy()

        return goals

    def ft_get_short_term_actions(self,
                                  goals,
                                  mode,
                                  radius
                                  ):
        self.info = self.ft_info
        actions = []
        current_agent_pos = self.info["current_agent_pos"]
        for agent_id in range(self.num_agents):
            if self.use_constrict_map:
                self.ft_agent_plan_explored[agent_id] = np.maximum(self.ft_agent_plan_explored[agent_id],(self.ft_info['explored_each_map'][agent_id] > 0).astype(np.int32))
                self.ft_agent_plan_obstacle[agent_id] = np.maximum(self.ft_agent_plan_obstacle[agent_id],(self.ft_info['obstacle_each_map'][agent_id] > 0).astype(np.int32))
                for b in range(agent_id+1,self.num_agents):
                    rel_dis = self.distance(current_agent_pos[agent_id] ,current_agent_pos[b])
                    if rel_dis < 8:
                        self.ft_agent_plan_explored[agent_id] = np.maximum(self.ft_agent_plan_explored[agent_id],(self.ft_info['explored_each_map'][b] > 0).astype(np.int32))
                        self.ft_agent_plan_obstacle[agent_id] = np.maximum(self.ft_agent_plan_obstacle[agent_id],(self.ft_info['obstacle_each_map'][b] > 0).astype(np.int32))
                explored = self.ft_agent_plan_explored[agent_id]
                obstacle = self.ft_agent_plan_obstacle[agent_id]

            elif self.use_merge_plan:
                explored = (self.info['explored_all_map'] > 0).astype(np.int32)
                obstacle = (self.info['obstacle_all_map'] > 0).astype(np.int32)
            else:
                explored = (self.info['explored_each_map'][agent_id] > 0).astype(np.int32)
                obstacle = (self.info['obstacle_each_map'][agent_id] > 0).astype(np.int32)
            if self.use_agent_obstacle:
                for a in range(self.num_agents):
                    if a != agent_id:
                        obstacle[current_agent_pos[a][0], current_agent_pos[a][1]] = 1

            H, W = explored.shape
            map = np.ones((H, W)).astype(np.int32) * 3  # 3 for unknown area
            map[explored == 1] = 0  # 0 for explored area
            map[obstacle == 1] = 1  # 1 for obstacles
            map[:self.agent_view_size, :] = 1
            map[H-self.agent_view_size:, :] = 1
            map[:, :self.agent_view_size] = 1
            map[:, W-self.agent_view_size:] = 1
            # Set unexplored.
            unexplored = (map == 3).astype(np.int32)
            # Initialize cost map.
            temp_map = map.copy().astype(np.float32)
            temp_map[map != 1] = 1  # free & frontiers & unknown
            temp_map[map == 1] = np.inf  # obstacles

            if mode == 'normal':
                pass
            elif mode == 'utility':
                # cost = 1 - unexplored (%)
                H, W = map.shape
                for x in range(H):
                    for y in range(W):
                        if map[x, y] == 1:
                            temp_map[x, y] = np.inf
                        else:
                            utility = unexplored[x-radius:x+radius+1, y-radius:y +
                                                radius+1].sum() / (math.pow(radius*2+1, 2))
                            temp_map[x, y] = 1.0 + (1.0 - utility) * 2.0
            else:
                raise NotImplementedError

            goal = [goals[agent_id][0], goals[agent_id][1]]
            agent_pos = [current_agent_pos[agent_id][0], current_agent_pos[agent_id][1]]
            agent_dir = self.agent_dir[agent_id]
            path = pyastar2d.astar_path(temp_map, agent_pos, goal, allow_diagonal=False)
            if type(path) == type(None) or len(path) == 1:
                actions.append(1)
                continue
            relative_pos = np.array(path[1]) - np.array(agent_pos)
            action = self.relative_pose2action(agent_dir, relative_pos)
            actions.append(action)

        return actions

    def relative_pose2action(self, agent_dir, relative_pos):
        # first quadrant
        if relative_pos[0] < 0 and relative_pos[1] > 0:
            if agent_dir == 0 or agent_dir == 3:
                return 2  # forward
            if agent_dir == 1:
                return 0  # turn left
            if agent_dir == 2:
                return 1  # turn right
        # second quadrant
        if relative_pos[0] > 0 and relative_pos[1] > 0:
            if agent_dir == 0 or agent_dir == 1:
                return 2  # forward
            if agent_dir == 2:
                return 0  # turn left
            if agent_dir == 3:
                return 1  # turn right
        # third quadrant
        if relative_pos[0] > 0 and relative_pos[1] < 0:
            if agent_dir == 1 or agent_dir == 2:
                return 2  # forward
            if agent_dir == 3:
                return 0  # turn left
            if agent_dir == 0:
                return 1  # turn right
        # fourth quadrant
        if relative_pos[0] < 0 and relative_pos[1] < 0:
            if agent_dir == 2 or agent_dir == 3:
                return 2  # forward
            if agent_dir == 0:
                return 0  # turn left
            if agent_dir == 1:
                return 1  # turn right
        if relative_pos[0] == 0 and relative_pos[1] == 0:
            # turn around
            return 1
        if relative_pos[0] == 0 and relative_pos[1] > 0:
            if agent_dir == 0:
                return 2
            if agent_dir == 1:
                return 0
            else:
                return 1
        if relative_pos[0] == 0 and relative_pos[1] < 0:
            if agent_dir == 2:
                return 2
            if agent_dir == 1:
                return 1
            else:
                return 0
        if relative_pos[0] > 0 and relative_pos[1] == 0:
            if agent_dir == 1:
                return 2
            if agent_dir == 0:
                return 1
            else:
                return 0
        if relative_pos[0] < 0 and relative_pos[1] == 0:
            if agent_dir == 3:
                return 2
            if agent_dir == 0:
                return 0
            else:
                return 1
        return None
    def distance(self,pos_a,pos_b):
        dis = np.square(pos_a[0]-pos_b[0])+np.square(pos_a[1]-pos_b[1])
        return dis

    def get_short_term_action(self, inputs):
        actions = []
        for agent_id in range(self.num_agents):
            if self.use_constrict_map:
                self.agent_plan_explored[agent_id] = np.maximum(self.agent_plan_explored[agent_id],(self.ft_info['explored_each_map'][agent_id] > 0).astype(np.int32)[
                    self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height])
                self.agent_plan_obstacle[agent_id] = np.maximum(self.agent_plan_obstacle[agent_id],(self.ft_info['obstacle_each_map'][agent_id] > 0).astype(np.int32)[
                    self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height])
                for b in range(agent_id+1,self.num_agents):
                    rel_dis = self.distance(self.agent_pos[agent_id] ,self.agent_pos[b])
                    if rel_dis < 8:
                        self.agent_plan_explored[agent_id] = np.maximum(self.agent_plan_explored[agent_id],(self.ft_info['explored_each_map'][b] > 0).astype(np.int32)[
                            self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height])
                        self.agent_plan_obstacle[agent_id] = np.maximum(self.agent_plan_obstacle[agent_id],(self.ft_info['obstacle_each_map'][b] > 0).astype(np.int32)[
                            self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height])
                explored = self.agent_plan_explored[agent_id]
                obstacle = self.agent_plan_obstacle[agent_id]

            elif self.use_merge_plan:
                explored = (self.ft_info['explored_all_map'] > 0).astype(np.int32)[
                    self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
                obstacle = (self.ft_info['obstacle_all_map'] > 0).astype(np.int32)[
                    self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
            else:
                explored = (self.ft_info['explored_each_map'][agent_id] > 0).astype(np.int32)[
                    self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
                obstacle = (self.ft_info['obstacle_each_map'][agent_id] > 0).astype(np.int32)[
                    self.agent_view_size:self.agent_view_size+self.width, self.agent_view_size:self.agent_view_size+self.height]
            
            if self.use_agent_obstacle:
                for a in range(self.num_agents):
                    if a != agent_id:
                        obstacle[self.agent_pos[a][1], self.agent_pos[a][0]] = 1

            temp_map = np.ones((self.width, self.height), dtype=np.float32)
            temp_map[obstacle == 1] = 1000
            if self.astar_cost_mode == 'normal':
                pass
            elif self.astar_cost_mode == 'utility':
                # cost = 1 - unexplored%
                unexplored = 1 - explored
                radius = self.astar_utility_radius
                H, W = explored.shape
                for x in range(H):
                    for y in range(W):
                        if obstacle[x, y] == 1:
                            temp_map[x, y] = 1000
                        else:
                            left_x = x-radius
                            right_x = x+radius+1
                            left_y = y-radius
                            right_y = y+radius+1

                            if x < radius :
                                left_x=0
                            if y < radius :
                                left_y=0
                            if H-x < radius:
                                right_x=H
                            if W-y < radius:
                                right_y=W
                            
                            utility = unexplored[left_x:right_x, left_y:right_y].sum() / ((right_x-left_x)*(right_y-left_y))
                            temp_map[x, y] = 1.0 + (1.0 - utility) * 2.0
            else:
                raise NotImplementedError
            
            goal = [int(inputs[agent_id][1]), int(inputs[agent_id][0])]
            agent_pos = [self.agent_pos[agent_id][1], self.agent_pos[agent_id][0]]
            agent_dir = self.agent_dir[agent_id]
            path = pyastar2d.astar_path(temp_map, agent_pos, goal, allow_diagonal=False)
            if len(path) == 1:
                actions.append(1)
                continue
            relative_pos = np.array(path[1]) - np.array(agent_pos)
            # first quadrant
            if relative_pos[0] < 0 and relative_pos[1] > 0:
                if agent_dir == 0 or agent_dir == 3:
                    actions.append(2)  # forward
                    continue
                if agent_dir == 1:
                    actions.append(0)  # turn left
                    continue
                if agent_dir == 2:
                    actions.append(1)  # turn right
                    continue
            # second quadrant
            if relative_pos[0] > 0 and relative_pos[1] > 0:
                if agent_dir == 0 or agent_dir == 1:
                    actions.append(2)  # forward
                    continue
                if agent_dir == 2:
                    actions.append(0)  # turn left
                    continue
                if agent_dir == 3:
                    actions.append(1)  # turn right
                    continue
            # third quadrant
            if relative_pos[0] > 0 and relative_pos[1] < 0:
                if agent_dir == 1 or agent_dir == 2:
                    actions.append(2)  # forward
                    continue
                if agent_dir == 3:
                    actions.append(0)  # turn left
                    continue
                if agent_dir == 0:
                    actions.append(1)  # turn right
                    continue
            # fourth quadrant
            if relative_pos[0] < 0 and relative_pos[1] < 0:
                if agent_dir == 2 or agent_dir == 3:
                    actions.append(2)  # forward
                    continue
                if agent_dir == 0:
                    actions.append(0)  # turn left
                    continue
                if agent_dir == 1:
                    actions.append(1)  # turn right
                    continue
            if relative_pos[0] == 0 and relative_pos[1] == 0:
                # turn around
                actions.append(1)
                continue
            if relative_pos[0] == 0 and relative_pos[1] > 0:
                if agent_dir == 0:
                    actions.append(2)
                    continue
                if agent_dir == 1:
                    actions.append(0)
                    continue
                else:
                    actions.append(1)
                    continue
            if relative_pos[0] == 0 and relative_pos[1] < 0:
                if agent_dir == 2:
                    actions.append(2)
                    continue
                if agent_dir == 1:
                    actions.append(1)
                    continue
                else:
                    actions.append(0)
                    continue
            if relative_pos[0] > 0 and relative_pos[1] == 0:
                if agent_dir == 1:
                    actions.append(2)
                    continue
                if agent_dir == 0:
                    actions.append(1)
                    continue
                else:
                    actions.append(0)
                    continue
            if relative_pos[0] < 0 and relative_pos[1] == 0:
                if agent_dir == 3:
                    actions.append(2)
                    continue
                if agent_dir == 0:
                    actions.append(0)
                    continue
                else:
                    actions.append(1)
                    continue
        '''
        for i in range(self.num_agents):
            goal = inputs[i]
            agent_pos = self.agent_pos[i]
            agent_dir = self.agent_dir[i]
            relative_pos = np.array(goal) - np.array(agent_pos)
            # first quadrant
            if relative_pos[0] >= 0 and relative_pos[1] <= 0:
                if agent_dir == 0 or agent_dir == 3:
                    actions.append(2)  # forward
                    continue
                if  agent_dir == 1:
                    actions.append(0)  # turn left
                    continue
                if  agent_dir == 2:
                    actions.append(1)  # turn right
                    continue
            # second quadrant
            if relative_pos[0] >= 0 and relative_pos[1] >= 0:
                if agent_dir == 0 or agent_dir == 1:
                    actions.append(2)  # forward
                    continue
                if  agent_dir == 2:
                    actions.append(0)  # turn left
                    continue
                if  agent_dir == 3:
                    actions.append(1)  # turn right
                    continue
            # third quadrant
            if relative_pos[0] <= 0 and relative_pos[1] >= 0:
                if agent_dir == 1 or agent_dir == 2:
                    actions.append(2)  # forward
                    continue
                if  agent_dir == 3:
                    actions.append(0)  # turn left
                    continue
                if  agent_dir == 0:
                    actions.append(1)  # turn right
                    continue
            # fourth quadrant
            if relative_pos[0] <= 0 and relative_pos[1] <= 0:
                if agent_dir == 2 or agent_dir == 3:
                    actions.append(2)  # forward
                    continue
                if  agent_dir == 0:
                    actions.append(0)  # turn left
                    continue
                if  agent_dir == 1:
                    actions.append(1)  # turn right
                    continue
        '''
        return actions

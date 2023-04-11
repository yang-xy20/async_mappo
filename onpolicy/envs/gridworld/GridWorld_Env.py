import gym
from .gym_minigrid.envs.human import HumanEnv
from onpolicy.envs.gridworld.gym_minigrid.register import register
import numpy as np
from icecream import ic
from onpolicy.utils.multi_discrete import MultiDiscrete

class GridWorldEnv(object):
    def __init__(self, args):

        self.num_agents = args.num_agents
        self.scenario_name = args.scenario_name
        self.use_random_pos = args.use_random_pos
        self.agent_pos = None if self.use_random_pos else args.agent_pos
        self.num_obstacles = args.num_obstacles
        self.use_single_reward = args.use_single_reward
        self.use_discrect = args.use_discrect

        register(
            id=self.scenario_name,
            grid_size=args.grid_size,
            max_steps=args.max_steps,
            local_step_num=args.local_step_num,
            agent_view_size=args.agent_view_size,
            num_agents=self.num_agents,
            num_obstacles=self.num_obstacles,
            agent_pos=self.agent_pos,
            use_merge_plan=args.use_merge_plan,
            use_merge=args.use_merge,
            use_constrict_map=args.use_constrict_map,
            use_fc_net=args.use_fc_net,
            use_agent_id=args.use_agent_id,
            use_stack=args.use_stack,
            use_orientation=args.use_orientation,
            use_same_location=args.use_same_location,
            use_complete_reward=args.use_complete_reward,
            use_agent_obstacle=args.use_agent_obstacle,
            use_multiroom=args.use_multiroom,
            use_irregular_room=args.use_irregular_room,
            use_time_penalty=args.use_time_penalty,
            use_overlap_penalty=args.use_overlap_penalty,
            entry_point='onpolicy.envs.gridworld.gym_minigrid.envs:MultiExplorationEnv',
            astar_cost_mode=args.astar_cost_mode
        )

        self.env = gym.make(self.scenario_name)
        self.max_steps = self.env.max_steps
        # print("max step is {}".format(self.max_steps))

        self.observation_space = self.env.observation_space
        self.share_observation_space = self.env.observation_space
        
        if self.use_discrect:
            self.action_space = [
                MultiDiscrete([[0, args.grid_size - 1],[0, args.grid_size - 1]])
                for _ in range(self.num_agents)
            ]
        else:
            self.action_space = [
            gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
            for _ in range(self.num_agents)
            ]

    def seed(self, seed=None):
        if seed is None:
            self.env.seed(1)
        else:
            self.env.seed(seed)

    def reset(self, choose=True):
        if choose:
            obs, info = self.env.reset()
        else:
            obs = [
                {
                    'image': np.zeros((self.env.width, self.env.height, 3), dtype='uint8'),
                    'direction': 0,
                    'mission': " "
                } for agent_id in range(self.num_agents)
            ]
            info = {}
        return obs, info

    def step(self, actions):
        if not np.all(actions == np.ones((self.num_agents, 1)).astype(np.int) * (-1.0)):
            obs, rewards, done, infos = self.env.step(actions)
            dones = np.array([done for agent_id in range(self.num_agents)])
            if self.use_single_reward:
                rewards = 0.3 * np.expand_dims(infos['agent_explored_reward'], axis=1) + 0.7 * np.expand_dims(
                    np.array([infos['merge_explored_reward'] for _ in range(self.num_agents)]), axis=1)
            else:
                rewards = np.expand_dims(
                    np.array([infos['merge_explored_reward'] for _ in range(self.num_agents)]), axis=1)
        else:
            obs = [
                {
                    'image': np.zeros((self.env.width, self.env.height, 3), dtype='uint8'),
                    'direction': 0,
                    'mission': " "
                } for agent_id in range(self.num_agents)
            ]
            rewards = np.zeros((self.num_agents, 1))
            dones = np.array([None for agent_id in range(self.num_agents)])
            infos = {}

        return obs, rewards, dones, infos

    def close(self):
        self.env.close()

    def get_short_term_action(self, input):
        outputs = self.env.get_short_term_action(input)
        return outputs

    def render(self, mode="human", short_goal_pos=None):
        if mode == "human":
            self.env.render(mode=mode, short_goal_pos=short_goal_pos)
        else:
            return self.env.render(mode=mode, short_goal_pos=short_goal_pos)

    def ft_get_short_term_goals(self, args, mode=""):
        mode_list = ['apf', 'utility', 'nearest', 'rrt', 'voronoi']
        assert mode in mode_list, (f"frontier global mode should be in {mode_list}")
        return self.env.ft_get_short_term_goals(args, mode=mode)

    def ft_get_short_term_actions(self, *args):
        return self.env.ft_get_short_term_actions(*args)

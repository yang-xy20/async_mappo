import json
import time
import wandb
import os
import copy
import numpy as np
from itertools import chain
import torch
import imageio
from icecream import ic
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict, deque
from onpolicy.utils.util import update_linear_schedule, get_shape_from_act_space, AsynchControl
from onpolicy.runner.shared.base_runner import Runner
import torch.nn as nn


def _t2n(x):
    return x.detach().cpu().numpy()


class GridWorldRunner(Runner):
    def __init__(self, config):
        super(GridWorldRunner, self).__init__(config)
        self.init_hyperparameters()
        self.init_map_variables()
        self.init_keys()

        if self.asynch:
            def generate_random_period():
                return np.random.randint(3, 5+1)
            self.asynch_control = AsynchControl(num_envs=self.n_rollout_threads, num_agents=self.num_agents, limit=self.episode_length, random_fn=generate_random_period, min_length=5, max_length=5)

    def run(self):
        start = time.time()
        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.warmup()
        episodes = int(self.num_env_steps) // self.max_steps // self.n_rollout_threads
        
        for episode in range(episodes):
            self.init_env_info()
            
            self.init_map_variables()
            period_rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1))
            auc_area = np.zeros((self.n_rollout_threads, self.max_steps), dtype=np.float32)
            auc_single_area = np.zeros((self.n_rollout_threads, self.num_agents, self.max_steps), dtype=np.float32)

            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)


            for step in range(self.max_steps):

                local_step = step % self.local_step_num
                global_step = (step // self.local_step_num) % self.episode_length

                actions_env = self.envs.get_short_term_action(self.short_term_goal)

                # Obser reward and next obs
                dict_obs, rewards, dones, infos = self.envs.step(actions_env)
                for e in range(self.n_rollout_threads):
                    for key in self.sum_env_info_keys:
                        if key in infos[e].keys():
                            self.env_info['sum_{}'.format(key)][e] += np.array(infos[e][key])
                    for key in self.equal_env_info_keys:
                        if key == 'merge_explored_ratio':
                            auc_area[e, step] = np.array(infos[e][key])
                            if np.all(dones[e]):
                                self.env_info[key][e] = infos[e][key]
                        if key == 'agent_explored_ratio':
                            auc_single_area[e, :, step] = np.array(infos[e][key])
                            if np.all(dones[e]):
                                self.env_info[key][e] = infos[e][key]
                        elif key in infos[e].keys():
                            if key == 'explored_ratio_step':
                                for agent_id in range(self.num_agents):
                                    agent_k = "agent{}_{}".format(agent_id, key)
                                    if agent_k in infos[e].keys():
                                        self.env_info[key][e][agent_id] = infos[e][agent_k]
                            else:
                                self.env_info[key][e] = infos[e][key]
                    if step in [49, 99,149,199,249,299]:
                        self.env_info[str(step+1)+'step_merge_auc'][e] = auc_area[e, :step+1].sum().copy()
                        self.env_info[str(step+1)+'step_auc'][e] = auc_single_area[e, :, :step+1].sum(axis = 1)
                
                period_rewards += rewards
                if self.asynch:
                    self.asynch_control.step()
                    if step == self.max_steps - 1:
                        self.asynch_control.reset()
                if (not self.asynch and local_step == self.local_step_num - 1) or (self.asynch and np.any(self.asynch_control.active)):

                    # For every global step, update the full and local maps
                    data = dict_obs, period_rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                    # insert data into buffer
                    if not self.asynch:
                        self.insert(data)
                        period_rewards = np.zeros((self.n_rollout_threads, self.num_agents, 1))
                    else:
                        self.insert(data, active_agents=self.asynch_control.active_agents())
                        for e, a, s in self.asynch_control.active_agents():
                            period_rewards[e, a, 0] = 0.

                    current_agent_pos = np.array([infos[e]['current_agent_pos']
                                                 for e in range(self.n_rollout_threads)])
                    current_agent_pos -= self.agent_view_size
                    for e in range(self.n_rollout_threads):
                        for a in range(self.num_agents):
                            self.lmb[e, a] = self.get_local_map_boundaries(
                                current_agent_pos[e, a], (self.local_size, self.local_size), (self.map_size, self.map_size))
                    
                    if not self.asynch:
                        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.compute_global_goal(
                            step=global_step + 1)
                    else:
                        par_values, par_actions, par_action_log_probs, par_rnn_states, par_rnn_states_critic = self.async_compute_global_goal(self.asynch_control.active_agents())
                        active_mask = (self.asynch_control.active == 1)
                        values[active_mask] = par_values
                        actions[active_mask] = par_actions
                        action_log_probs[active_mask] = par_action_log_probs
                        rnn_states[active_mask] = par_rnn_states
                        rnn_states_critic[active_mask] = par_rnn_states_critic

            # compute return and update network
            if self.asynch:
                self.buffer.update_mask(self.asynch_control.cnt)
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.max_steps * self.n_rollout_threads

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            self.convert_info()
            print("average episode merge explored reward is {}".format(np.mean(self.env_infos['sum_merge_explored_reward'])))
            print("average episode merge explored ratio is {}".format(np.mean(self.env_infos['merge_explored_ratio'])))
            
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                self.log_env(self.env_infos, total_num_steps)
                self.log_agent(self.env_infos, total_num_steps)
                self.log_train(train_infos, total_num_steps)
                
            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
               self.eval(total_num_steps)

    def _eval_convert(self, dict_obs, infos):
        obs = {}
        obs['image'] = np.zeros((len(dict_obs), self.num_agents, self.full_w-2 *
                                self.agent_view_size, self.full_h-2*self.agent_view_size, 3), dtype=np.float32)
        if self.use_agent_id:
            if self.use_fc_net:
                obs['vector'] = np.zeros((len(dict_obs), self.num_agents,
                                        self.num_agents), dtype=np.float32)
            else:
                obs['vector_cnn'] = np.zeros((len(dict_obs), self.num_agents, self.num_agents+1, self.full_w -
                                            2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
        obs['global_obs'] = np.zeros((len(dict_obs), self.num_agents, 4, self.full_w -
                                     2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
        if self.use_orientation:
            if self.use_merge:
                obs['global_direction'] = np.zeros(
                    (len(dict_obs), self.num_agents, self.num_agents, 4), dtype=np.float32)
            else:
                obs['global_direction'] = np.zeros(
                    (len(dict_obs), self.num_agents, 1, 4), dtype=np.float32)
        agent_pos_map = np.zeros((len(dict_obs), self.num_agents,
                                 self.full_w, self.full_h), dtype=np.float32)
        if self.use_merge:
            # obs['global_merge_goal'] = np.zeros((len(dict_obs), self.num_agents, 2, self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
            obs['global_merge_obs'] = np.zeros((len(dict_obs), self.num_agents, 4, self.full_w -
                                               2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
            merge_pos_map = np.zeros((len(dict_obs), self.full_w, self.full_h), dtype=np.float32)
            # eval_global_merge_goal = np.zeros((len(dict_obs), self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
        for e in range(len(dict_obs)):
            if len(infos[e]) == 0:
                continue
            for agent_id in range(self.num_agents):
                agent_pos_map[e, agent_id, infos[e]['current_agent_pos'][agent_id][0],
                              infos[e]['current_agent_pos'][agent_id][1]] = (agent_id + 1) * self.augment[agent_id]
                self.eval_all_agent_pos_map[e, agent_id, infos[e]['current_agent_pos'][agent_id]
                                            [0], infos[e]['current_agent_pos'][agent_id][1]] = (agent_id + 1) * self.augment[agent_id]
                if self.use_merge:
                    '''a = int(self.short_term_goal[e][agent_id][0])
                    b = int(self.short_term_goal[e][agent_id][1])

                    if eval_global_merge_goal[e, a, b] != (agent_id + 1) * self.augment and\
                    eval_global_merge_goal[e, a, b] != np.array([agent_id + 1 for agent_id in range(self.num_agents)]).sum() * self.augment:
                        eval_global_merge_goal[e, a, b] += (agent_id + 1) * self.augment

                    if self.eval_global_merge_goal_trace[e, a, b] != (agent_id + 1) * self.augment and\
                    self.eval_global_merge_goal_trace[e, a, b] != np.array([agent_id + 1 for agent_id in range(self.num_agents)]).sum() * self.augment:
                        self.eval_global_merge_goal_trace[e, a, b] += (agent_id + 1) * self.augment'''
                    if self.use_agent_id:
                        merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0], infos[e]
                                    ['current_agent_pos'][agent_id][1]] += (agent_id + 1) * self.augment[agent_id]
                        
                        self.eval_all_merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0],
                                                    infos[e]['current_agent_pos'][agent_id][1]] += (agent_id + 1) * self.augment[agent_id]
                        if ((self.eval_all_merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0],
                                                    infos[e]['current_agent_pos'][agent_id][1]]) + ((agent_id + 1) * self.augment[agent_id])) <= np.array([agent_id + 1 for agent_id in range(self.num_agents)]).sum() * self.augment[agent_id]:
                            self.eval_all_merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0],
                                                    infos[e]['current_agent_pos'][agent_id][1]] += (agent_id + 1) * self.augment[agent_id]
                    else:
                        merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0], infos[e]
                                    ['current_agent_pos'][agent_id][1]] = self.augment[agent_id]
                        self.eval_all_merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0],
                                                    infos[e]['current_agent_pos'][agent_id][1]] = self.augment[agent_id]
                    

        for e in range(len(dict_obs)):
            if len(infos[e]) == 0:
                continue
            for agent_id in range(self.num_agents):
                obs['global_obs'][e, agent_id, 0] = infos[e]['explored_each_map'][agent_id][self.agent_view_size: self.full_w -
                                                                                            self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size] / 255.0
                obs['global_obs'][e, agent_id, 1] = infos[e]['obstacle_each_map'][agent_id][self.agent_view_size:self.full_w -
                                                                                            self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size] / 255.0
                obs['global_obs'][e, agent_id, 2] = agent_pos_map[e, agent_id][self.agent_view_size:self.full_w -
                                                                               self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size] / 255.0
                obs['global_obs'][e, agent_id, 3] = self.eval_all_agent_pos_map[e, agent_id][self.agent_view_size:self.full_w -
                                                                                             self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size] / 255.0
                # self.agent_local_map[e, agent_id] = infos[e]['agent_local_map'][agent_id]
                obs['image'][e, agent_id] = cv2.resize(infos[e]['agent_local_map'][agent_id], (
                    self.full_w - 2*self.agent_view_size, self.full_h - 2*self.agent_view_size))
                if self.use_merge:
                    # obs['global_merge_goal'][e, agent_id, 0] = eval_global_merge_goal[e]
                    # obs['global_merge_goal'][e, agent_id, 1] = self.eval_global_merge_goal_trace[e]
                    obs['global_merge_obs'][e, agent_id, 0] = infos[e]['explored_all_map'][self.agent_view_size:self.full_w -
                                                                                           self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size] / 255.0
                    obs['global_merge_obs'][e, agent_id, 1] = infos[e]['obstacle_all_map'][self.agent_view_size:self.full_w -
                                                                                           self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size] / 255.0
                    obs['global_merge_obs'][e, agent_id, 2] = merge_pos_map[e][self.agent_view_size:self.full_w -
                                                                               self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size] / 255.0
                    obs['global_merge_obs'][e, agent_id, 3] = self.eval_all_merge_pos_map[e][self.agent_view_size:self.full_w -
                                                                                             self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size] / 255.0
                if self.use_agent_id:
                    if self.use_fc_net:
                        obs['vector'][e, agent_id] = np.eye(self.num_agents)[agent_id]
                    else:
                        obs['vector_cnn'][e, agent_id, 0] = np.ones((self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size)) * (
                            agent_id+1) / np.array([aa+1 for aa in range(self.num_agents)]).sum()
                if self.use_orientation:
                    i = 0
                    obs['global_direction'][e, agent_id, i] = np.eye(
                        4)[infos[e]['agent_direction'][agent_id]]
                    if self.use_merge:
                        for l in range(self.num_agents):
                            if l != agent_id:
                                i += 1
                                obs['global_direction'][e, agent_id, i] = np.eye(
                                    4)[infos[e]['agent_direction'][l]]

            if self.use_agent_id:
                if not (self.use_fc_net or self.use_stack):
                    obs['vector_cnn'][e, :, 1:] = np.expand_dims(
                        obs['global_obs'][e, :, 2], 1).repeat(self.num_agents, axis=1)
        if self.use_stack:
            all_global_cnn_input = [[] for _ in range(self.num_agents)]
            for agent_id in range(self.num_agents):
                for key in obs.keys():
                    if key not in ['stack_obs','global_direction', 'vector']:
                        if key == 'image':
                            all_global_cnn_input[agent_id].append(obs[key][:, agent_id].transpose((0,3,1,2))/ 255.0)
                        else:
                            all_global_cnn_input[agent_id].append(obs[key][:, agent_id])
                all_global_cnn_input[agent_id] = np.concatenate(all_global_cnn_input[agent_id], axis=1) #[e,n,...]
            
            all_global_cnn_input = np.stack(all_global_cnn_input, axis=1)
            obs = {}
            obs['stack_obs'] = np.zeros((len(dict_obs), self.num_agents, all_global_cnn_input.shape[2] * self.num_agents, self.full_w -
                                     2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
            for agent_id in range(self.num_agents):
                obs['stack_obs'][:, agent_id] = all_global_cnn_input.reshape(len(dict_obs), -1, *all_global_cnn_input.shape[3:]).copy()
            
            for a in range(1, self.num_agents):
                obs['stack_obs'][:, a] = np.roll(obs['stack_obs'][:,a], -all_global_cnn_input.shape[2]*a, axis=1)

        if self.visualize_input:
            self.visualize_obs(self.fig, self.ax, obs)

        return obs

    def _convert(self, dict_obs, infos):
        obs = {}
        obs['image'] = np.zeros((len(dict_obs), self.num_agents, self.full_w-2 *
                                self.agent_view_size, self.full_h-2*self.agent_view_size, 3), dtype=np.float32)
        if self.use_agent_id:
            if self.use_fc_net:
                obs['vector'] = np.zeros((len(dict_obs), self.num_agents,
                                        self.num_agents), dtype=np.float32)
            elif self.use_stack:
                obs['vector_cnn'] = np.zeros((len(dict_obs), self.num_agents, 1, self.full_w -
                                            2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
            else:
                obs['vector_cnn'] = np.zeros((len(dict_obs), self.num_agents, 1+self.num_agents, self.full_w -
                                            2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)

        obs['global_obs'] = np.zeros((len(dict_obs), self.num_agents, 4, self.full_w -
                                     2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
        if self.use_orientation:
            if self.use_merge:
                obs['global_direction'] = np.zeros(
                    (len(dict_obs), self.num_agents, self.num_agents, 4), dtype=np.float32)
                # obs['global_merge_goal'] = np.zeros((len(dict_obs), self.num_agents, 2, self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
            else:
                obs['global_direction'] = np.zeros(
                    (len(dict_obs), self.num_agents, 1, 4), dtype=np.float32)
        agent_pos_map = np.zeros((len(dict_obs), self.num_agents,
                                 self.full_w, self.full_h), dtype=np.float32)
        if self.use_merge:
            obs['global_merge_obs'] = np.zeros((len(dict_obs), self.num_agents, 4, self.full_w -
                                               2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
            merge_pos_map = np.zeros((len(dict_obs), self.full_w, self.full_h), dtype=np.float32)
            self.merge_map = np.zeros((len(dict_obs), self.full_w, self.full_h), dtype=np.float32)
            # global_merge_goal = np.zeros((len(dict_obs), self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)

        for e in range(len(dict_obs)):
            for agent_id in range(self.num_agents):
                agent_pos_map[e, agent_id, infos[e]['current_agent_pos'][agent_id][0],
                              infos[e]['current_agent_pos'][agent_id][1]] = (agent_id + 1) * self.augment[agent_id]
                self.all_agent_pos_map[e, agent_id, infos[e]['current_agent_pos'][agent_id][0],
                                       infos[e]['current_agent_pos'][agent_id][1]] = (agent_id + 1) * self.augment[agent_id]
                if self.use_merge:
                    
                    '''a = int(self.short_term_goal[e][agent_id][0])
                    b = int(self.short_term_goal[e][agent_id][1])
                    if global_merge_goal[e, a, b] != (agent_id + 1) * self.augment and\
                    global_merge_goal[e, a, b] != np.array([agent_id + 1 for agent_id in range(self.num_agents)]).sum() * self.augment:
                        global_merge_goal[e, a, b] += (agent_id + 1) * self.augment

                    if self.global_merge_goal_trace[e, a, b] != (agent_id + 1) * self.augment and\
                    self.global_merge_goal_trace[e, a, b] != np.array([agent_id + 1 for agent_id in range(self.num_agents)]).sum() * self.augment:
                        self.global_merge_goal_trace[e, a, b] += (agent_id + 1) * self.augment'''
                   
                    if self.use_agent_id:
                        merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0], infos[e]
                                  ['current_agent_pos'][agent_id][1]] += (agent_id + 1) * self.augment[agent_id]
                        
                        self.all_merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0],
                                                    infos[e]['current_agent_pos'][agent_id][1]] += (agent_id + 1) * self.augment[agent_id]
                        if ((self.all_merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0],
                                                    infos[e]['current_agent_pos'][agent_id][1]]) + ((agent_id + 1) * self.augment[agent_id])) <= np.array([agent_id + 1 for agent_id in range(self.num_agents)]).sum() * self.augment[agent_id]:
                            self.all_merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0],
                                                    infos[e]['current_agent_pos'][agent_id][1]] += (agent_id + 1) * self.augment[agent_id]
                    else:
                        merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0], infos[e]
                                    ['current_agent_pos'][agent_id][1]] = self.augment[agent_id]
                        self.all_merge_pos_map[e, infos[e]['current_agent_pos'][agent_id][0],
                                                    infos[e]['current_agent_pos'][agent_id][1]] = self.augment[agent_id]

        for e in range(len(dict_obs)):
            self.merge_map = infos[e]['explored_all_map'][self.agent_view_size:self.full_w -
                                                          self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size] / 255.0
            for agent_id in range(self.num_agents):
                obs['global_obs'][e, agent_id, 0] = infos[e]['explored_each_map'][agent_id][self.agent_view_size: self.full_w -
                                                                                            self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size] / 255.0
                obs['global_obs'][e, agent_id, 1] = infos[e]['obstacle_each_map'][agent_id][self.agent_view_size:self.full_w -
                                                                                            self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size] / 255.0
                obs['global_obs'][e, agent_id, 2] = agent_pos_map[e, agent_id][self.agent_view_size:self.full_w -
                                                                               self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size] / 255.0
                obs['global_obs'][e, agent_id, 3] = self.all_agent_pos_map[e, agent_id][self.agent_view_size:self.full_w -
                                                                                        self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size] / 255.0
                # self.agent_local_map[e, agent_id] = infos[e]['agent_local_map'][agent_id]
                obs['image'][e, agent_id] = cv2.resize(infos[e]['agent_local_map'][agent_id], (
                    self.full_w - 2*self.agent_view_size, self.full_h - 2*self.agent_view_size))
                if self.use_merge:
                    # obs['global_merge_goal'][e, agent_id, 0] = global_merge_goal[e]
                    # obs['global_merge_goal'][e, agent_id, 1] = self.global_merge_goal_trace[e]
                    obs['global_merge_obs'][e, agent_id, 0] = infos[e]['explored_all_map'][self.agent_view_size:self.full_w -
                                                                                           self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size] / 255.0
                    obs['global_merge_obs'][e, agent_id, 1] = infos[e]['obstacle_all_map'][self.agent_view_size:self.full_w -
                                                                                           self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size] / 255.0
                    obs['global_merge_obs'][e, agent_id, 2] = merge_pos_map[e][self.agent_view_size:self.full_w -
                                                                               self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size] / 255.0
                    obs['global_merge_obs'][e, agent_id, 3] = self.all_merge_pos_map[e][self.agent_view_size:self.full_w -
                                                                                self.agent_view_size, self.agent_view_size:self.full_w-self.agent_view_size] / 255.0
                
                if self.use_agent_id: 
                    if self.use_fc_net:
                        obs['vector'][e, agent_id] = np.eye(self.num_agents)[agent_id]
                    else:
                        obs['vector_cnn'][e, agent_id, 0] = np.ones((self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size)) * (
                            agent_id+1) / np.array([aa+1 for aa in range(self.num_agents)]).sum()

                if self.use_orientation:
                    i = 0
                    obs['global_direction'][e, agent_id, i] = np.eye(
                        4)[infos[e]['agent_direction'][agent_id]]
                    if self.use_merge:
                        for l in range(self.num_agents):
                            if l != agent_id:
                                i += 1
                                obs['global_direction'][e, agent_id, i] = np.eye(
                                    4)[infos[e]['agent_direction'][l]]
            if self.use_agent_id:
                if not (self.use_fc_net or self.use_stack):
                    obs['vector_cnn'][e, :, 1:] = np.expand_dims(
                        obs['global_obs'][e, :, 2], 1).repeat(self.num_agents, axis=1)

        if self.use_stack:
            all_global_cnn_input = [[] for _ in range(self.num_agents)]
            for agent_id in range(self.num_agents):
                for key in obs.keys():
                    if key not in ['stack_obs','global_direction', 'vector']:
                        if key == 'image':
                            all_global_cnn_input[agent_id].append(obs[key][:, agent_id].transpose((0,3,1,2))/ 255.0)
                        else:
                            all_global_cnn_input[agent_id].append(obs[key][:, agent_id])
                all_global_cnn_input[agent_id] = np.concatenate(all_global_cnn_input[agent_id], axis=1) #[e,n,...]
            
            all_global_cnn_input = np.stack(all_global_cnn_input, axis=1)
            obs = {}
            obs['stack_obs'] = np.zeros((len(dict_obs), self.num_agents, all_global_cnn_input.shape[2] * self.num_agents, self.full_w -
                                     2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
            for agent_id in range(self.num_agents):
                obs['stack_obs'][:, agent_id] = all_global_cnn_input.reshape(len(dict_obs), -1, *all_global_cnn_input.shape[3:]).copy()
            
            for a in range(1, self.num_agents):
                obs['stack_obs'][:, a] = np.roll(obs['stack_obs'][:,a], -all_global_cnn_input.shape[2]*a, axis=1)

        if self.visualize_input:
            self.visualize_obs(self.fig, self.ax, obs)

        return obs

    def warmup(self):
        # reset env
        dict_obs, info = self.envs.reset()

        obs = self._convert(dict_obs, info)
        self.obs = obs
        # if not self.use_centralized_V:
        share_obs = self._convert(dict_obs, info)

        for key in obs.keys():
            self.buffer.obs[key][0] = obs[key].copy()

        for key in share_obs.keys():
            self.buffer.share_obs[key][0] = share_obs[key].copy()

        current_agent_pos = np.array([info[e]['current_agent_pos']
                                     for e in range(self.n_rollout_threads)])
        current_agent_pos -= self.agent_view_size

        for e in range(self.n_rollout_threads):
            for a in range(self.num_agents):
                self.lmb[e, a] = self.get_local_map_boundaries(
                    current_agent_pos[e, a], (self.local_size, self.local_size), (self.map_size, self.map_size))

        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.compute_global_goal(
            0)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic


    def init_hyperparameters(self):
        # Calculating full and local map sizes
        self.map_size = self.all_args.grid_size
        self.max_steps = self.all_args.max_steps
        self.local_size = self.all_args.local_size
        self.local_step_num = self.all_args.local_step_num
        self.agent_view_size = self.all_args.agent_view_size
        self.full_w, self.full_h = self.map_size + 2 * \
            self.agent_view_size, self.map_size + 2*self.agent_view_size
        self.asynch = self.all_args.asynch
        self.use_stack = self.all_args.use_stack
        self.use_agent_id = self.all_args.use_agent_id

        # grid goal
        self.grid_goal = self.all_args.grid_goal
        self.goal_grid_size = self.all_args.goal_grid_size

        # function_parameters
        self.use_merge = self.all_args.use_merge
        self.use_intrinsic_reward = self.all_args.use_intrinsic_reward
        self.use_global_goal = self.all_args.use_global_goal
        self.use_orientation = self.all_args.use_orientation
        self.use_fc_net = self.all_args.use_fc_net
        self.visualize_input = self.all_args.visualize_input
        self.use_up_agents = self.all_args.use_up_agents
        self.up_agents_step = self.all_args.up_agents_step
        self.use_down_agents = self.all_args.use_down_agents
        self.down_agents_step = self.all_args.down_agents_step
        self.use_discrect = self.all_args.use_discrect

        # eval by time_step
        self.max_timestep = self.all_args.max_timestep

        if self.use_agent_id:
            self.augment = [255 // (np.array([agent_id+1 for agent_id in range(self.num_agents)]).sum()) for _ in range(self.num_agents)]
        else:
            self.augment = [255 // (agent_id+1) for agent_id in range(self.num_agents)]
        self.best_gobal_reward = -np.inf

        if self.visualize_input:
            plt.ion()
            self.fig, self.ax = plt.subplots(
                self.num_agents*3, 4, figsize=(10, 2.5), facecolor="whitesmoke")
    
    def init_keys(self):
        # info keys
        self.equal_env_info_keys = ['agent_explored_ratio', 'merge_explored_ratio','merge_explored_ratio_step', 'merge_explored_ratio_step_0.98','explored_ratio_step',
        'merge_overlap_ratio','50step_merge_auc','100step_merge_auc','50step_auc','100step_auc','150step_merge_auc','200step_merge_auc','250step_merge_auc','300step_merge_auc','150step_auc','200step_auc','250step_auc','300step_auc']
        self.sum_env_info_keys = ['merge_explored_reward', 'agent_explored_reward']

        #log keys
        self.agents_env_info_keys = ['agent_explored_ratio','sum_agent_explored_reward', 'explored_ratio_step', '50step_auc','100step_auc','150step_auc','200step_auc','250step_auc','300step_auc']
        self.env_info_keys = ['merge_explored_ratio','sum_merge_explored_reward','merge_overlap_ratio', 'merge_explored_ratio_step','merge_explored_ratio_step_0.98', '50step_merge_auc','100step_merge_auc','150step_merge_auc','200step_merge_auc','250step_merge_auc','300step_merge_auc']
            
        if self.use_eval:
            self.eval_env_info_keys = ['eval_merge_explored_ratio','eval_merge_explored_ratio_step','eval_50step_merge_auc','eval_100step_merge_auc','eval_150step_merge_auc','eval_200step_merge_auc','eval_merge_explored_ratio_step_0.98','eval_merge_overlap_ratio']
            self.auc_infos_keys = ['merge_auc','agent_auc']

        # convert keys
        self.env_infos_keys = self.agents_env_info_keys + self.env_info_keys + \
                        ['max_merge_explored_ratio','min_merge_explored_ratio','merge_success_rate'] 

        self.env_infos = {}
        for key in self.env_infos_keys:
            self.env_infos[key] = deque(maxlen=1)
    
    def init_env_info(self):
        self.env_info = {}

        for key in self.agents_env_info_keys:
            if "step" in key:
                self.env_info[key] = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32) * self.max_steps
            else:
                self.env_info[key] = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        
        for key in self.env_info_keys:
            if "step" in key:
                self.env_info[key] = np.ones((self.n_rollout_threads,), dtype=np.float32) * self.max_steps
            else:
                self.env_info[key] = np.zeros((self.n_rollout_threads,), dtype=np.float32)
    
    def init_eval_env_info(self):
        self.eval_env_info = {}
        for key in self.eval_env_info_keys:
            if "step" in key:
                self.eval_env_info[key] = np.ones((self.n_eval_rollout_threads,), dtype=np.float32) * self.max_steps
            else:
                self.eval_env_info[key] = np.zeros((self.n_eval_rollout_threads,), dtype=np.float32)

    def init_map_variables(self):
        # Initializing full, merge and local map
        # self.global_goal = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.float32)
        self.short_term_goal = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.int)
        self.all_agent_pos_map = np.zeros(
            (self.n_rollout_threads, self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        # Local Map Boundaries
        self.lmb = np.zeros((self.n_rollout_threads, self.num_agents, 4)).astype(int)

        if self.use_merge:
            # self.global_merge_goal_trace = np.zeros((self.n_rollout_threads, self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
            self.all_merge_pos_map = np.zeros(
                (self.n_rollout_threads, self.full_w, self.full_h), dtype=np.float32)

    def init_eval_map_variables(self):
        # Initializing full, merge and local map
        self.eval_all_agent_pos_map = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        if self.use_merge:
            # self.eval_global_merge_goal_trace = np.zeros((self.n_eval_rollout_threads, self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
            self.eval_all_merge_pos_map = np.zeros(
                (self.n_eval_rollout_threads, self.full_w, self.full_h), dtype=np.float32)

    def get_local_map_boundaries(self, agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes
        gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
        gx2, gy2 = gx1 + local_w, gy1 + local_h
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w

        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h

        return [gx1, gx2, gy1, gy2]
    
    def save_global_model(self, step):
        if len(self.env_infos["sum_merge_explored_reward"]) >= self.all_args.eval_episodes and \
            (np.mean(self.env_infos["sum_merge_explored_reward"]) >= self.best_gobal_reward):
            self.best_gobal_reward = np.mean(self.env_infos["sum_merge_explored_reward"])
            torch.save(self.trainer.policy.actor.state_dict(), str(self.save_dir) + "/global_actor_best.pt")
            torch.save(self.trainer.policy.critic.state_dict(), str(self.save_dir) + "/global_critic_best.pt")
            torch.save(self.trainer.policy.actor_optimizer.state_dict(), str(self.save_dir) + "/global_actor_optimizer_best.pt")
            torch.save(self.trainer.policy.critic_optimizer.state_dict(), str(self.save_dir) + "/global_critic_optimizer_best.pt")  
        torch.save(self.trainer.policy.actor.state_dict(), str(self.save_dir) + "/global_actor_periodic_{}.pt".format(step))
        torch.save(self.trainer.policy.critic.state_dict(), str(self.save_dir) + "/global_critic_periodic_{}.pt".format(step))
        torch.save(self.trainer.policy.actor_optimizer.state_dict(), str(self.save_dir) + "/global_actor_optimizer_periodic_{}.pt".format(step))
        torch.save(self.trainer.policy.critic_optimizer.state_dict(), str(self.save_dir) + "/global_critic_optimizer_periodic_{}.pt".format(step))

    def convert_info(self):
        for k, v in self.env_info.items():
            if k == "explored_ratio_step":
                self.env_infos[k].append(v)
                for agent_id in range(self.num_agents):
                    print("agent{}_{}: {}/{}".format(agent_id, k, np.mean(v[:, agent_id]), self.max_steps))
                print('minimal agent {}: {}/{}'.format(k, np.min(v), self.max_steps))
            elif k == "merge_explored_ratio_step":
                self.env_infos['merge_success_rate'].append((v != self.max_steps).sum() / self.n_rollout_threads)
                v_copy = v.copy()
                v_copy[v == self.max_steps] = np.nan
                self.env_infos[k].append(v)
                print('mean valid {}: {}'.format(k, np.nanmean(v_copy)))
            else:
                self.env_infos[k].append(v)
                if k == 'merge_explored_ratio':       
                    self.env_infos['max_merge_explored_ratio'].append(np.max(v))
                    self.env_infos['min_merge_explored_ratio'].append(np.min(v))
                    print(np.mean(v))

    def convert_eval_info(self):
        for k, v in self.eval_env_info.items():
            if k == "eval_merge_explored_ratio_step":
                self.eval_env_infos[k].append(v)
            else:
                self.eval_env_infos[k].append(v)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.nanmean(v) if k == "merge_explored_ratio_step" or k == "merge_explored_ratio_step_0.98" else np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.nanmean(v) if k == "merge_explored_ratio_step" or k == "merge_explored_ratio_step_0.98" else np.mean(v)}, total_num_steps)

    def log_agent(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if "merge" not in k:
                for agent_id in range(self.num_agents):
                    agent_k = "agent{}_".format(agent_id) + k
                    if self.use_wandb:
                        wandb.log({agent_k: np.mean(np.array(v)[:,:,agent_id])}, step=total_num_steps)
                    else:
                        self.writter.add_scalars(agent_k, {agent_k: np.mean(np.array(v)[:,:,agent_id])}, total_num_steps)

    @torch.no_grad()
    def compute_global_goal(self, step):
        self.trainer.prep_rollout()

        concat_share_obs = {}
        concat_obs = {}

        for key in self.buffer.share_obs.keys():
            concat_share_obs[key] = np.concatenate(self.buffer.share_obs[key][step])
        for key in self.buffer.obs.keys():
            concat_obs[key] = np.concatenate(self.buffer.obs[key][step])

        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(concat_share_obs,
                                              concat_obs,
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        if self.all_args.grid_goal:
            action = action.detach().clone()
            # print("pre action", action)
            action[:, 1:3] = nn.Sigmoid()(action[:, 1:3])
            action = np.array(np.split(_t2n(action), self.n_rollout_threads))
            # print("action", action)
            r, c = action[:, :, 0].astype(
                np.int32) // self.goal_grid_size, action[:, :, 0].astype(np.int32) % self.goal_grid_size
            short_term_goal = np.zeros(
                (self.n_rollout_threads, self.num_agents, 2), dtype=np.float32)
            short_term_goal[:, :, 0] = (action[:, :, 1] + r) / self.goal_grid_size
            short_term_goal[:, :, 1] = (action[:, :, 2] + c) / self.goal_grid_size
            # print("short term goal", short_term_goal)
        elif self.use_discrect:
            short_term_goal = np.array(np.split(_t2n(action), self.n_rollout_threads))
        else:
            short_term_goal = np.array(np.split(_t2n(nn.Sigmoid()(action)), self.n_rollout_threads))
        
        if self.use_global_goal:
            if self.use_discrect:
                self.short_term_goal = short_term_goal.astype(np.int)
            else:
                self.short_term_goal = (short_term_goal * self.map_size).astype(np.int)
        else:
            self.short_term_goal = (short_term_goal * self.local_size).astype(np.int)
            for e in range(self.n_rollout_threads):
                for a in range(self.num_agents):
                    self.short_term_goal[e, a, 1] += self.lmb[e, a, 0]
                    self.short_term_goal[e, a, 0] += self.lmb[e, a, 2]
        
        self.short_term_goal[self.short_term_goal>=self.map_size]=self.map_size-1
        self.short_term_goal[self.short_term_goal<0]=0

        return values, actions, action_log_probs, rnn_states, rnn_states_critic
    
    @torch.no_grad()
    def async_compute_global_goal(self, active_agents):
        self.trainer.prep_rollout()

        concat_share_obs = {}
        concat_obs = {}

        for key in self.buffer.share_obs.keys():
            concat_share_obs[key] = np.stack([self.buffer.share_obs[key][step, e, a] for e, a, step in active_agents], axis=0)
        for key in self.buffer.obs.keys():
            concat_obs[key] = np.stack([self.buffer.obs[key][step, e, a] for e, a, step in active_agents], axis=0)

        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(concat_share_obs,
                                              concat_obs,
                                              np.stack([self.buffer.rnn_states[step, e, a] for e, a, step in active_agents], axis=0),
                                              np.stack([self.buffer.rnn_states_critic[step, e, a] for e, a, step in active_agents], axis=0),
                                              np.stack([self.buffer.masks[step, e, a] for e, a, step in active_agents], axis=0))
        # [self.envs, agents, dim]
        values = _t2n(value)
        actions = _t2n(action)
        action_log_probs = _t2n(action_log_prob)
        rnn_states = _t2n(rnn_states)
        rnn_states_critic = _t2n(rnn_states_critic)

        if self.all_args.grid_goal:
            action = action.detach().clone()
            # print("pre action", action)
            action[:, 1:3] = nn.Sigmoid()(action[:, 1:3])
            action = _t2n(action)
            # print("action", action)
            r, c = action[:, 0].astype(
                np.int32) // self.goal_grid_size, action[:, 0].astype(np.int32) % self.goal_grid_size
            short_term_goal = np.zeros(
                (len(active_agents), 2), dtype=np.float32)
            short_term_goal[:, 0] = (action[:, 1] + r) / self.goal_grid_size
            short_term_goal[:, 1] = (action[:, 2] + c) / self.goal_grid_size
            # print("short term goal", short_term_goal)
        elif self.use_discrect:
            short_term_goal = _t2n(action)
        else:
            short_term_goal = _t2n(nn.Sigmoid()(action))
    
        if self.use_global_goal:
            if self.use_discrect:
                short_term_goal = short_term_goal.astype(np.int)
            else:
                short_term_goal = (short_term_goal * self.map_size).astype(np.int)
        else:
            short_term_goal = (short_term_goal * self.local_size).astype(np.int)
            for i, (e, a, s) in enumerate(active_agents):
                short_term_goal[i, 1] += self.lmb[e, a, 0]
                short_term_goal[i, 0] += self.lmb[e, a, 2]
        for i, (e, a, s) in enumerate(active_agents):
            short_term_goal[i, 0] = min(short_term_goal[i, 0], self.map_size-1)
            short_term_goal[i, 1] = min(short_term_goal[i, 1], self.map_size-1)
            self.short_term_goal[e, a] = short_term_goal[i]
        self.short_term_goal[self.short_term_goal>=self.map_size]=self.map_size-1
        self.short_term_goal[self.short_term_goal<0]=0
        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def eval_compute_global_goal(self, infos, use_ft):
        if self.all_args.algorithm_name in ["rmappo", "mappo", "rmappg", "mappg"]:
            self.trainer.prep_rollout()
        if self.asynch and not self.all_args.use_time:
            self.asynch_control.step()
        current_agent_pos = np.array([infos[e]['current_agent_pos']
                                     for e in range(self.n_rollout_threads)])
        current_agent_pos -= self.agent_view_size
        for e in range(self.n_rollout_threads):
            for a in range(self.num_agents):
                self.lmb[e, a] = self.get_local_map_boundaries(
                    current_agent_pos[e, a], (self.local_size, self.local_size), (self.map_size, self.map_size))
        concat_obs = {}
        for key in self.obs.keys():
            concat_obs[key] = np.concatenate(self.obs[key])

        # Get short-term goals.
        if use_ft:
            
            self.ft_short_term_goals = self.envs.ft_get_short_term_goals(
                self.all_args, mode=self.all_args.algorithm_name[3:]
            )
            # Used to render for ft methods.
            if (not self.asynch) or self.all_args.use_time:
                self.short_term_goals = np.array([
                    [
                        # (x, y) ---> (y, x) in minigrid
                        (goal[1] - self.agent_view_size, goal[0] - self.agent_view_size)
                        for goal in env_goals
                    ]
                    for env_goals in self.ft_short_term_goals
                ])
            else:
                short_term_goals = [
                    [
                        # (x, y) ---> (y, x) in minigrid
                        (goal[1] - self.agent_view_size, goal[0] - self.agent_view_size)
                        for goal in env_goals
                    ]
                    for env_goals in self.ft_short_term_goals
                ]
                if not hasattr(self, 'short_term_goals'):
                        self.short_term_goals = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.int)
                self.short_term_goals = (short_term_goals * self.asynch_control.active.reshape(self.n_rollout_threads, self.num_agents, 1)).astype(np.int) \
                    + (self.short_term_goals * (1-self.asynch_control.active.reshape(self.n_rollout_threads, self.num_agents, 1))).astype(np.int)
            
            self.short_term_goals[self.short_term_goals>=self.map_size]=self.map_size-1
            self.short_term_goals[self.short_term_goals<0]=0

        elif self.all_args.algorithm_name in ["rmappo", "mappo", "rmappg", "mappg"]:
            action, rnn_states = self.trainer.policy.act(
                concat_obs,
                np.concatenate(self.rnn_states),
                np.concatenate(self.masks),
                deterministic=True
            )
            self.rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

            if self.all_args.grid_goal:
                action = action.detach().clone()
                action[:, 1:3] = nn.Sigmoid()(action[:, 1:3])
                action = np.array(np.split(_t2n(action), self.n_rollout_threads))
                r, c = action[:, :, 0].astype(
                    np.int32) // self.goal_grid_size, action[:, :, 0].astype(np.int32) % self.goal_grid_size
                short_term_goals = np.zeros(
                    (self.n_rollout_threads, self.num_agents, 2), dtype=np.float32)
                short_term_goals[:, :, 0] = (action[:, :, 1] + r) / self.goal_grid_size
                short_term_goals[:, :, 1] = (action[:, :, 2] + c) / self.goal_grid_size
            elif self.use_discrect:
                short_term_goals = np.array(
                    np.split(_t2n(action), self.n_rollout_threads))
            else:
                short_term_goals = np.array(
                    np.split(_t2n(nn.Sigmoid()(action)), self.n_rollout_threads))

            if (not self.asynch) or self.all_args.use_time:
                if self.use_global_goal:
                    if self.use_discrect:
                        self.short_term_goals = short_term_goals.astype(np.int)
                    else:
                        self.short_term_goals = (
                            short_term_goals * self.map_size).astype(np.int)
                else:
                    self.short_term_goals = (
                        short_term_goals * self.local_size).astype(np.int)
                    for e in range(self.n_rollout_threads):
                        for a in range(self.num_agents):
                            self.short_term_goals[e, a, 1] += self.lmb[e, a, 0]
                            self.short_term_goals[e, a, 0] += self.lmb[e, a, 2]
                self.short_term_goals[self.short_term_goals>=self.map_size]=self.map_size-1
                self.short_term_goals[self.short_term_goals<0]=0
            else:
                if self.use_global_goal:
                    if self.use_discrect:
                        short_term_goals = short_term_goals.astype(np.int)
                    else:
                        short_term_goals = (short_term_goals * self.map_size).astype(np.int)
                else:
                    short_term_goals = (short_term_goals * self.local_size).astype(np.int)
                    for e in range(self.n_rollout_threads):
                        for a in range(self.num_agents):
                            short_term_goals[e, a, 1] += self.lmb[e, a, 0]
                            short_term_goals[e, a, 0] += self.lmb[e, a, 2]
                for e in range(self.n_rollout_threads):
                    for a in range(self.num_agents):
                        short_term_goals[e, a, 0] = min(short_term_goals[e, a, 0], self.map_size-1)
                        short_term_goals[e, a, 1] = min(short_term_goals[e, a, 1], self.map_size-1)
                if not hasattr(self, 'short_term_goals'):
                    self.short_term_goals = np.zeros((self.n_rollout_threads, self.num_agents, 2), dtype=np.int)
                self.short_term_goals = (short_term_goals * self.asynch_control.active.reshape(self.n_rollout_threads, self.num_agents, 1)).astype(np.int) \
                    + (self.short_term_goals * (1-self.asynch_control.active.reshape(self.n_rollout_threads, self.num_agents, 1))).astype(np.int)
                self.short_term_goals[self.short_term_goals>=self.map_size]=self.map_size-1
                self.short_term_goals[self.short_term_goals<0]=0
    
    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()

        concat_share_obs = {}
        for key in self.buffer.share_obs.keys():
            concat_share_obs[key] = np.concatenate(self.buffer.share_obs[key][-1])

        next_values = self.trainer.policy.get_values(concat_share_obs,
                                                     np.concatenate(
                                                         self.buffer.rnn_states_critic[-1]),
                                                     np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def insert(self, data, active_agents=None):
        dict_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data
        dones_env = np.all(dones, axis=-1)

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(
        ), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        obs = self._convert(dict_obs, infos)
        self.obs = obs
        share_obs = self._convert(dict_obs, infos)

        self.all_agent_pos_map[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.full_w, self.full_h), dtype=np.float32)
        if self.use_merge:
            self.all_merge_pos_map[dones_env == True] = np.zeros(
                ((dones_env == True).sum(), self.full_w, self.full_h), dtype=np.float32)
            # self.global_merge_goal_trace[dones_env == True] = np.zeros(((dones_env == True).sum(), self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
        if self.use_intrinsic_reward:
            for e in range(self.n_rollout_threads):
                for agent_id in range(self.num_agents):
                    if self.merge_map[e][self.short_term_goal[e, agent_id, 1], self.short_term_goal[e, agent_id, 0]] == 0:
                        rewards[e, agent_id, 0] += 0.01                    

        if active_agents is None:
            self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)
        else:
            self.buffer.async_insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, active_agents=active_agents)

    def visualize_obs(self, fig, ax, obs):
        # individual
        for agent_id in range(self.num_agents * 3):
            sub_ax = ax[agent_id]
            for i in range(4):
                sub_ax[i].clear()
                sub_ax[i].set_yticks([])
                sub_ax[i].set_xticks([])
                sub_ax[i].set_yticklabels([])
                sub_ax[i].set_xticklabels([])
                if agent_id < self.num_agents:
                    sub_ax[i].imshow(obs['global_obs'][0, agent_id, i])
                elif agent_id < self.num_agents*2 and self.use_merge:
                    sub_ax[i].imshow(obs['global_merge_obs'][0, agent_id-self.num_agents, i])
                # elif i<2: sub_ax[i].imshow(obs['global_merge_goal'][0, agent_id-self.num_agents*2, i])
                # elif i < 5:
                    # sub_ax[i].imshow(obs['global_merge_goal'][0, agent_id-self.num_agents, i-4])
                    # sub_ax[i].imshow(obs['gt_map'][0, agent_id - self.num_agents, i-4])
        plt.gcf().canvas.flush_events()
        # plt.pause(0.1)
        fig.canvas.start_event_loop(0.001)
        plt.gcf().canvas.flush_events()

    @torch.no_grad()
    def eval(self, total_num_steps):

        action_shape = get_shape_from_act_space(self.eval_envs.action_space[0])
        eval_episode_rewards = []
        self.eval_env_infos = defaultdict(list)
        self.init_eval_env_info()
        reset_choose = np.ones(self.n_eval_rollout_threads) == 1.0
        self.init_eval_map_variables()
        eval_dict_obs, eval_infos = self.eval_envs.reset(reset_choose)
        eval_obs = self._eval_convert(eval_dict_obs, eval_infos)

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        eval_dones_env = np.zeros(self.n_eval_rollout_threads, dtype=bool)
        self.eval_lmb = np.zeros((self.n_eval_rollout_threads, self.num_agents, 4)).astype(int)
        local_step = 0
        eval_auc_area = np.zeros((self.n_eval_rollout_threads, self.max_steps), dtype=np.float32)

        while True:
            eval_choose = (eval_dones_env == False)
            if ~np.any(eval_choose):
                break
            if self.grid_goal:
                eval_actions = np.ones(
                    (self.n_eval_rollout_threads, self.num_agents, 3)).astype(np.int) * (-1.0)
            else:
                eval_actions = np.ones(
                    (self.n_eval_rollout_threads, self.num_agents, action_shape)).astype(np.int) * (-1.0)
            short_term_goal = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, action_shape), dtype=np.float32)
            self.short_term_goal = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, action_shape)).astype(np.int)

            self.trainer.prep_rollout()

            if local_step % self.local_step_num == 0:
                current_agent_pos = np.array([eval_infos[e]['current_agent_pos']
                                             for e in range(self.n_eval_rollout_threads)])
                current_agent_pos -= self.agent_view_size
                for e in range(self.n_eval_rollout_threads):
                    for a in range(self.num_agents):
                        self.eval_lmb[e, a] = self.get_local_map_boundaries(
                            current_agent_pos[e, a], (self.local_size, self.local_size), (self.map_size, self.map_size))

                concat_eval_obs = {}
                for key in eval_obs.keys():
                    concat_eval_obs[key] = np.concatenate(eval_obs[key][eval_choose])
                eval_action, eval_rnn_state = self.trainer.policy.act(concat_eval_obs,
                                                                      np.concatenate(
                                                                          eval_rnn_states[eval_choose]),
                                                                      np.concatenate(
                                                                          eval_masks[eval_choose]),
                                                                      deterministic=True)

                eval_actions[eval_choose] = np.array(
                    np.split(_t2n(eval_action), (eval_choose == True).sum()))
                eval_rnn_states[eval_choose] = np.array(
                    np.split(_t2n(eval_rnn_state), (eval_choose == True).sum()))

                # Obser reward and next obs
                if self.all_args.grid_goal:
                    eval_action = eval_action.detach().clone()
                    # print("pre action", action)
                    eval_action[:, 1:3] = nn.Sigmoid()(eval_action[:, 1:3])
                    eval_action = np.array(np.split(_t2n(eval_action), (eval_choose == True).sum()))
                    # print("action", action)
                    r, c = eval_action[:, :, 0].astype(
                        np.int32) // self.goal_grid_size, eval_action[:, :, 0].astype(np.int32) % self.goal_grid_size
                    eval_action[:, :, 1] = (eval_action[:, :, 1] + r) / self.goal_grid_size
                    eval_action[:, :, 2] = (eval_action[:, :, 2] + c) / self.goal_grid_size
                    eval_action = eval_action[:, :, 1:3]
                    short_term_goal[eval_choose] = eval_action
                else:
                    short_term_goal[eval_choose] = np.array(
                        np.split(_t2n(nn.Sigmoid()(eval_action)), (eval_choose == True).sum()))
                if self.use_global_goal:
                    self.short_term_goal[eval_choose] = (
                        short_term_goal[eval_choose] * self.map_size).astype(np.int)
                else:
                    self.short_term_goal[eval_choose] = (
                        short_term_goal[eval_choose] * self.local_size).astype(np.int)
                    for e in range(self.n_eval_rollout_threads):
                        for a in range(self.num_agents):
                            self.short_term_goal[e, a, 1] += self.eval_lmb[e, a, 0]
                            self.short_term_goal[e, a, 0] += self.eval_lmb[e, a, 2]
                self.short_term_goal[self.short_term_goal>=self.map_size]=self.map_size-1
                self.short_term_goal[self.short_term_goal<0]=0

            eval_actions_env = self.eval_envs.get_short_term_action(self.short_term_goal)
            # Obser reward and next obs
            eval_dict_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(
                eval_actions_env)
            
            for e in range(self.n_eval_rollout_threads):
                for key in eval_infos[e].keys():
                    if key == 'merge_explored_ratio':
                        eval_auc_area[e, local_step] = np.array(eval_infos[e][key])
                        if np.all(eval_dones[e]):
                            self.eval_env_info['eval_'+key][e] = eval_infos[e][key]
                    elif 'eval_'+key in self.eval_env_info_keys:
                        self.eval_env_info['eval_'+key][e] = eval_infos[e][key]
                if local_step in [49, 99]:
                    self.eval_env_info['eval_'+str(local_step+1)+'step_merge_auc'][e] = eval_auc_area[e, :local_step+1].sum().copy()

            local_step += 1
            # if local_step % self.local_step_num == self.local_step_num -1:
            eval_obs = self._eval_convert(eval_dict_obs, eval_infos)
            eval_dones_env = np.all(eval_dones, axis=-1)

            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(
            ), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads,
                                 self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
            self.eval_all_agent_pos_map[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, self.full_w, self.full_h), dtype=np.float32)
            if self.use_merge:
                # self.eval_global_merge_goal_trace[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.full_w-2*self.agent_view_size, self.full_h-2*self.agent_view_size), dtype=np.float32)
                self.eval_all_merge_pos_map[eval_dones_env == True] = np.zeros(
                    ((eval_dones_env == True).sum(), self.full_w, self.full_h), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)

        self.convert_eval_info()
        print("eval average merge explored ratio is: " +
              str(np.mean(self.eval_env_infos['eval_merge_explored_ratio'])))
        self.log_env(self.eval_env_infos, total_num_steps)
                

    @torch.no_grad()
    def render(self):
        envs = self.envs
        # Init env infos.
        self.eval_infos = defaultdict(list)
        use_ft = self.all_args.algorithm_name[:2] == "ft"
        all_frames = []
        all_local_frames = []

        for episode in range(self.all_args.render_episodes):
            self.init_env_info()
            ic(episode)
            self.init_map_variables()
            auc_area = np.zeros((self.n_rollout_threads, self.max_steps), dtype=np.float32)
            auc_single_area = np.zeros((self.n_rollout_threads, self.num_agents, self.max_steps), dtype=np.float32)
            self.rnn_states = np.zeros((self.n_rollout_threads, self.num_agents,self.recurrent_N, self.hidden_size),dtype=np.float32)
            self.masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            reset_choose = np.ones(self.n_rollout_threads) == 1.0
            dict_obs, infos = envs.reset(reset_choose)
            obs = self._convert(dict_obs, infos)
            self.obs = obs
            self.eval_compute_global_goal(infos, use_ft)
            episode_rewards = []
            
            if self.asynch:
                self.asynch_control.reset()
        
            for step in range(self.max_steps):
                calc_start = time.time()
                local_step = step % self.local_step_num                    
                            
                if use_ft:
                    actions_env = envs.ft_get_short_term_actions(
                        self.ft_short_term_goals,
                        self.all_args.astar_cost_mode,
                        self.all_args.astar_utility_radius
                    )
                else:
                    actions_env = envs.get_short_term_action(self.short_term_goals)
                # Take actions and observe reward and next obs.
                if step <= self.up_agents_step and self.use_up_agents:
                    actions_env = np.array(actions_env)
                    for a in range(self.use_up_agents):
                        actions_env[:,a]= np.ones((self.n_rollout_threads))*-2
                if step >= self.down_agents_step and self.use_down_agents:
                    actions_env = np.array(actions_env)
                    for a in range(self.use_down_agents):
                        actions_env[:,a]= np.ones((self.n_rollout_threads))*-2

                dict_obs, rewards, dones, infos = envs.step(actions_env)
                
                for e in range(self.n_rollout_threads):
                    for key in self.sum_env_info_keys:
                        if key in infos[e].keys():
                            self.env_info['sum_{}'.format(key)][e] += np.array(infos[e][key])
                    for key in self.equal_env_info_keys:
                        if key == 'merge_explored_ratio':
                            auc_area[e, step] = np.array(infos[e][key])
                            if np.all(dones[e]):
                                self.env_info[key][e] = infos[e][key]
                        if key == 'agent_explored_ratio':
                            auc_single_area[e, :, step] = np.array(infos[e][key])
                            if np.all(dones[e]):
                                self.env_info[key][e] = infos[e][key]
                        elif key in infos[e].keys():
                            if key == 'explored_ratio_step':
                                for agent_id in range(self.num_agents):
                                    agent_k = "agent{}_{}".format(agent_id, key)
                                    if agent_k in infos[e].keys():
                                        self.env_info[key][e][agent_id] = infos[e][agent_k]
                            else:
                                self.env_info[key][e] = infos[e][key]
                    if step in [49, 99,149,199]:
                        self.env_info[str(step+1)+'step_merge_auc'][e] = auc_area[e, :step+1].sum().copy()
                        self.env_info[str(step+1)+'step_auc'][e] = auc_single_area[e, :, :step+1].sum(axis = 1)
                
                if local_step == self.local_step_num - 1:
                    obs = self._convert(dict_obs, infos)
                    self.obs = obs
                    episode_rewards.append(rewards)
                    dones_env = np.all(dones, axis=-1)
                    self.rnn_states[dones_env == True] = np.zeros(
                        ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                    self.masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                    self.masks[dones_env == True] = np.zeros(
                        ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
                    self.eval_compute_global_goal(infos, use_ft)
                
                if self.use_render: #and episode == 0:
                    if self.all_args.save_gifs:
                        image, local_image = envs.render('rgb_array', self.short_term_goals)[0]
                        all_frames.append(image)
                        all_local_frames.append(local_image)
                        calc_end = time.time()
                        elapsed = calc_end - calc_start
                        if elapsed < self.all_args.ifi:
                            time.sleep(self.all_args.ifi - elapsed)
                    else:
                        envs.render('human', self.short_term_goals)
            
            self.convert_info()
            total_num_steps = (episode + 1) * self.max_steps * self.n_rollout_threads
            if not self.use_render :
                self.log_env(self.env_infos, total_num_steps)
                self.log_agent(self.env_infos, total_num_steps)
            
        for k, v in self.env_infos.items():
            print("eval average {}: {}".format(k, np.nanmean(v) if k == 'merge_explored_ratio_step' or k == "merge_explored_ratio_step_0.98"else np.mean(v)))

        if self.all_args.save_gifs:
            ic("rendering....")
            imageio.mimsave(str(self.gif_dir) + '/merge.gif',
                            all_frames, duration=self.all_args.ifi)
            imageio.mimsave(str(self.gif_dir) + '/local.gif',
                            all_local_frames, duration=self.all_args.ifi)
            ic("done")
    
    @torch.no_grad()
    def render_by_time(self):
        envs = self.envs
        # Init env infos.
        self.eval_infos = defaultdict(list)
        use_ft = self.all_args.algorithm_name[:2] == "ft"
        all_frames = []
        all_local_frames = []
        self.merge_ratio = np.zeros((self.n_rollout_threads,),dtype=np.float32)
        for episode in range(self.all_args.render_episodes):
            self.init_env_info()
            ic(episode)
            self.init_map_variables()
            auc_area = np.zeros((self.n_rollout_threads,), dtype=np.float32)
            auc_single_area = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32)
            self.rnn_states = np.zeros((self.n_rollout_threads, self.num_agents,self.recurrent_N, self.hidden_size),dtype=np.float32)
            self.masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            reset_choose = np.ones(self.n_rollout_threads) == 1.0
            dict_obs, infos = envs.reset(reset_choose)
            obs = self._convert(dict_obs, infos)
            self.obs = obs
            self.eval_compute_global_goal(infos, use_ft)
            episode_rewards = []

            cur_time = 0.
            agent_time = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32)
            step_action = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.int32) * (-2)
            local_step = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.int32)
            total_step = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.int32)
            swap_count = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.int32)
            action_delay = {0: 0.5, 1: 0.5, 2: 1.} # delay of turning left/right is 0.5s, delay of forward is 1s
            inference_delay = 0.1 # delay of inference is 0.1s
            last_infos = copy.deepcopy(infos)

            for step in [49, 99,149,199]:
                self.env_info[str(step+1)+'step_merge_auc'] = -np.ones_like(self.env_info[str(step+1)+'step_merge_auc'], dtype=np.float32)
                self.env_info[str(step+1)+'step_auc'] = -np.ones_like(self.env_info[str(step+1)+'step_auc'], dtype=np.float32)
            for key in ['merge_explored_ratio_step', 'merge_explored_ratio_step_0.98']:
                self.env_info[key] = self.max_timestep * np.ones_like(self.env_info[key], dtype=np.float32)

            while cur_time < self.max_timestep:
                calc_start = time.time()                 
                            
                if use_ft:
                    actions_env = envs.ft_get_short_term_actions(
                        self.ft_short_term_goals,
                        self.all_args.astar_cost_mode,
                        self.all_args.astar_utility_radius
                    )
                else:
                    actions_env = envs.get_short_term_action(self.short_term_goals)
                actions_env = np.array(actions_env)
                
                actions = np.ones_like(actions_env) * (-2)
                for e in range(self.n_rollout_threads):
                    for a in range(self.num_agents):
                        if agent_time[e, a] <= 1e-3:
                            # action delay done, could go on next action
                            actions[e, a] = actions_env[e, a]
                            agent_time[e, a] = action_delay[actions[e, a]]
                            local_step[e, a] += 1
                            total_step[e, a] += 1
                        else:
                            # delay not done yet
                            actions[e, a] = -2
                
                if self.asynch:
                    passed_time = agent_time.min()
                    agent_time -= passed_time
                    cur_time += passed_time
                else:
                    passed_time = agent_time.max()
                    agent_time = np.zeros_like(agent_time)
                    cur_time += passed_time
                # print(cur_time, actions, local_step, total_step, swap_count, last_infos[0]['merge_explored_ratio'], last_infos[0]['current_agent_pos'], self.short_term_goals)

                # Take actions and observe reward and next obs.
                for e in range(self.n_rollout_threads):
                    if self.merge_ratio[e]  <= 0.5 and self.use_up_agents:
                        for a in range(self.use_up_agents):
                            actions[e,a] = -2
                    if self.merge_ratio[e] >= 0.5 and self.use_down_agents:
                        for a in range(self.use_down_agents):
                            actions[e,a] = -2

                dict_obs, rewards, dones, infos = envs.step(actions)
                
                for e in range(self.n_rollout_threads):
                    for key in self.sum_env_info_keys:
                        if key in last_infos[e].keys():
                            self.env_info['sum_{}'.format(key)][e] += np.array(last_infos[e][key]) * passed_time
                    for key in self.equal_env_info_keys:
                        if key == 'merge_explored_ratio':
                            auc_area[e] += np.array(last_infos[e][key]) * passed_time
                            self.merge_ratio[e] =infos[e][key]
                            if np.all(dones[e]):
                                self.merge_ratio[e] = np.zeros((self.n_rollout_threads,),dtype=np.float32)
                                self.env_info[key][e] = infos[e][key]
                        if key == 'agent_explored_ratio':
                            auc_single_area[e, :] += np.array(last_infos[e][key]) * passed_time
                            if np.all(dones[e]):
                                self.env_info[key][e] = infos[e][key]
                        elif key in infos[e].keys() and key not in ['merge_explored_ratio_step', 'merge_explored_ratio_step_0.98']:
                            if key == 'explored_ratio_step':
                                for agent_id in range(self.num_agents):
                                    agent_k = "agent{}_{}".format(agent_id, key)
                                    if agent_k in infos[e].keys():
                                        self.env_info[key][e][agent_id] = infos[e][agent_k]
                            else:
                                self.env_info[key][e] = infos[e][key]
                    for step in [49, 99,149,199]:
                        if cur_time >= step and self.env_info[str(step+1)+'step_merge_auc'][e] < 0:
                            self.env_info[str(step+1)+'step_merge_auc'][e] = auc_area[e]
                        if cur_time >= step and self.env_info[str(step+1)+'step_auc'][e].mean() < 0:
                            self.env_info[str(step+1)+'step_auc'][e] = auc_single_area[e]
                    if infos[e]['merge_explored_ratio'] >= 0.9 and self.env_info['merge_explored_ratio_step'][e] >= self.max_timestep - 1e-3:
                        self.env_info['merge_explored_ratio_step'][e] = min(self.env_info['merge_explored_ratio_step'][e], cur_time)
                    if infos[e]['merge_explored_ratio'] >= 0.98 and self.env_info['merge_explored_ratio_step_0.98'][e] >= self.max_timestep - 1e-3:
                        self.env_info['merge_explored_ratio_step_0.98'][e] = min(self.env_info['merge_explored_ratio_step_0.98'][e], cur_time)
                
                last_infos = copy.deepcopy(infos)

                obs = self._convert(dict_obs, infos)
                self.obs = obs
                episode_rewards.append(rewards)
                dones_env = np.all(dones, axis=-1)
                self.rnn_states[dones_env == True] = np.zeros(
                    ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                self.masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                self.masks[dones_env == True] = np.zeros(
                    ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
                pre_short_term_goals = copy.deepcopy(self.short_term_goals)
                self.eval_compute_global_goal(infos, use_ft)
                for e in range(self.n_rollout_threads):
                    for a in range(self.num_agents):
                        if local_step[e, a] == self.local_step_num:
                            # update the short term goal
                            local_step[e, a] = 0
                            agent_time[e, a] += inference_delay
                            swap_count[e, a] += 1
                        else:
                            self.short_term_goals[e, a] = pre_short_term_goals[e, a]
            
                if self.use_render: #and episode == 0:
                    if self.all_args.save_gifs:
                        image, local_image = envs.render('rgb_array', self.short_term_goals)[0]
                        all_frames.append(image)
                        all_local_frames.append(local_image)
                        calc_end = time.time()
                        elapsed = calc_end - calc_start
                        if elapsed < self.all_args.ifi:
                            time.sleep(self.all_args.ifi - elapsed)
                    else:
                        envs.render('human', self.short_term_goals)
            
            self.convert_info()
            total_num_steps = (episode + 1) * self.max_steps * self.n_rollout_threads
            if not self.use_render :
                self.log_env(self.env_infos, total_num_steps)
                self.log_agent(self.env_infos, total_num_steps)
            
        for k, v in self.env_infos.items():
            print("eval average {}: {}".format(k, np.nanmean(v) if k == 'merge_explored_ratio_step' or k == "merge_explored_ratio_step_0.98"else np.mean(v)))

        if self.all_args.save_gifs:
            ic("rendering....")
            imageio.mimsave(str(self.gif_dir) + '/merge.gif',
                            all_frames, duration=self.all_args.ifi)
            imageio.mimsave(str(self.gif_dir) + '/local.gif',
                            all_local_frames, duration=self.all_args.ifi)
            ic("done")

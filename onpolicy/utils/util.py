import glob
import os
import numpy as np
import math
import torch
import torch.nn as nn

class AsynchControl:
    def __init__(self, num_envs, num_agents, limit, random_fn, min_length, max_length):
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.limit = limit
        self.random_fn = random_fn
        self.min_length = min_length
        self.max_length = max_length

        self.reset()
    
    def reset(self):
        self.cnt = np.zeros((self.num_envs, self.num_agents), dtype=np.int32)
        self.rest = np.zeros((self.num_envs, self.num_agents), dtype=np.int32)
        self.active = np.ones((self.num_envs, self.num_agents), dtype=np.int32)
        for e in range(self.num_envs):
            for a in range(self.num_agents):
                self.rest[e, a] = self.random_fn() # the first step is unlimited
    
    def step(self):
        for e in range(self.num_envs):
            for a in range(self.num_agents):
                self.rest[e, a] -= 1
                self.active[e, a] = 0
                if self.rest[e, a] <= 0:
                    if self.cnt[e, a] < self.limit:
                        self.cnt[e, a] += 1
                        self.active[e, a] = 1
                        self.rest[e, a] = min(max(self.random_fn(), self.min_length), self.max_length)

    def active_agents(self):
        ret = []
        for e in range(self.num_envs):
            for a in range(self.num_agents):
                if self.active[e, a]:
                    ret.append((e, a, self.cnt[e, a]))
        return ret

def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)
        
def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    elif obs_space.__class__.__name__ == 'Dict':
        obs_shape = obs_space.spaces
    else:
        raise NotImplementedError
    return obs_shape

def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1  
    return act_shape


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c
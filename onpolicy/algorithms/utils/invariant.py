import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from onpolicy.algorithms.utils.vit import ViT, Attention, PreNorm, Transformer, CrossAttention, FeedForward
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import random

def get_position_embedding(pos, hidden_dim, device = torch.device("cpu")):
    scaled_time = 2 * torch.arange(hidden_dim / 2) / hidden_dim
    scaled_time = 10000 ** scaled_time
    scaled_time = pos / scaled_time
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=0).to(device)

def get_explicit_position_embedding(n_embed_input=10, n_embed_output=8 , device = torch.device("cpu")):

    return nn.Embedding(n_embed_input, n_embed_output).to(device)

class AlterEncoder(nn.Module):
    def __init__(self, num_grids, input_dim,  depth = 2, hidden_dim = 128, heads = 4, dim_head = 32, mlp_dim = 128, dropout = 0.):
        super().__init__()
        self.num_grids = num_grids
        self.hidden_dim = hidden_dim
        self.depth = depth       
        self.encode_actor = nn.Linear(input_dim, hidden_dim)
        self.encode_other = nn.Linear(input_dim, hidden_dim)
        self.last_cross_attn = nn.ModuleList([
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2 * heads * dim_head, bias = False),
            CrossAttention(hidden_dim, heads = heads, dim_head = dim_head, dropout = dropout),
            PreNorm(hidden_dim, FeedForward(hidden_dim, mlp_dim, dropout = dropout))
        ])
    
    def forward(self, data):
        x, others = data
        B = x.shape[0]
        # print("alter_attn", x.shape)
        x = self.encode_actor(x)
        all = [x,]
        for i, y in enumerate(others):
            y = self.encode_other(y)
            all.append(y)
        num_agents = len(all)
        out = torch.stack(all, dim = 1) # B x num_agents x 64 x D
        out = rearrange(out, "b n g d -> (b g) n d", b = B, n = num_agents, g = self.num_grids)
        norm, to_kv, cross_attn, ff= self.last_cross_attn
        out = norm(out)
        x = out[:, :1, :] # 64B x 1 x D
        others = out[:, 1:, :] # 64B x (n-1) x D
        if num_agents > 1:
            k, v = to_kv(others).chunk(2, dim=-1)
            out = cross_attn(x, k, v) + x # # 64B x 1 x D
        else:
            out = x
        out = ff(out) + out
        out = rearrange(out, " (b g) n d -> n b g d", b = B, n = 1, g = self.num_grids)[0]
        return out

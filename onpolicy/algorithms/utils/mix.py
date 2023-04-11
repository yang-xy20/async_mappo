import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from onpolicy.algorithms.utils.invariant import AlterEncoder
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from .util import init
from icecream import ic

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)

class MIXBase(nn.Module):
    def __init__(self, args, obs_shape, cnn_layers_params=None):
        super(MIXBase, self).__init__()

        self._use_orthogonal = args.use_orthogonal
        self._activation_id = args.activation_id
        self._use_maxpool2d = args.use_maxpool2d
        self.hidden_size = args.hidden_size
        self.grid_goal = args.grid_goal
        self.goal_grid_size = args.goal_grid_size
        self.use_stack = args.use_stack
        self.grid_size  = args.grid_size
        self.num_agents  = args.num_agents
        self.depth = args.attn_depth
        self.cnn_keys = []
        self.embed_keys = []
        self.mlp_keys = []
        self.n_cnn_input = 0
        self.n_embed_input = 0
        self.n_mlp_input = 0

        for key in obs_shape:
            if obs_shape[key].__class__.__name__ == 'Box':
                key_obs_shape = obs_shape[key].shape
                if len(key_obs_shape) == 3:
                    self.cnn_keys.append(key)
                else:
                    if "orientation" in key:
                        self.embed_keys.append(key)
                    else:
                        self.mlp_keys.append(key)
            else:
                raise NotImplementedError

        if len(self.cnn_keys) > 0:
            self.cnn = self._build_cnn_model(obs_shape, cnn_layers_params, self.hidden_size, self._use_orthogonal, self._activation_id, cnn_trans_layer = args.cnn_trans_layer)
        if len(self.embed_keys) > 0:
            self.embed = self._build_embed_model(obs_shape)
        if len(self.mlp_keys) > 0:
            self.mlp = self._build_mlp_model(obs_shape, self.hidden_size, self._use_orthogonal, self._activation_id)
        if self.use_stack:
            self.agent_invariant = self._build_invariant_model()

    def forward(self, x):
        out_x = x
        if len(self.cnn_keys) > 0:
            cnn_input = self._build_cnn_input(x)
            if self.use_stack:
                cnn_x = []
                for i in range(self.num_agents):
                    cnn_x.append(self.cnn(cnn_input[:, 0+self.split_channel*i:self.split_channel+self.split_channel*i]))
                out_x = torch.cat(cnn_x, dim=1)
                channels = out_x.shape[1] // self.num_agents
                actor = rearrange(out_x[:, 0:channels], 'b c h w -> b (h w) c')
                others = [rearrange(out_x[:, i*channels:(i+1)*channels], 'b c h w ->  b (h w) c') for i in range(1, self.num_agents)]
                out_x = [actor, others]
                out_x = self.agent_invariant(out_x)
            else:
                out_x = self.cnn(cnn_input)
            
            if self.grid_goal:
                out_x = [self.cnn_to_linear(out_x), self.cnn_to_grid(out_x)]
            else:
                out_x = self.cnn_to_linear(out_x)

        if len(self.embed_keys) > 0:
            embed_input = self._build_embed_input(x)
            embed_x = self.embed(embed_input.long()).view(embed_input.size(0), -1)
            out_x = torch.cat([out_x, embed_x], dim=1)

        if len(self.mlp_keys) > 0:
            mlp_input = self._build_mlp_input(x)
            mlp_x = self.mlp(mlp_input).view(mlp_input.size(0), -1)
            out_x = torch.cat([out_x, mlp_x], dim=1) # ! wrong

        return out_x

    def _build_cnn_model(self, obs_shape, cnn_layers_params, hidden_size, use_orthogonal, activation_id, cnn_trans_layer=""):
        
        if cnn_layers_params is None:
            cnn_layers_params = [(32, 8, 4, 0), (64, 4, 2, 0), (64, 3, 1, 0)]
        else:
            def _convert(params):
                output = []
                for l in params.split(' '):
                    output.append(tuple(map(int, l.split(','))))
                return output
            cnn_layers_params = _convert(cnn_layers_params)
            cnn_trans_layer = _convert(cnn_trans_layer)
        
        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        
        if self.use_stack:
            for key in self.cnn_keys:
                if key in ['stack_obs']:
                    self.n_cnn_input += int(obs_shape[key].shape[0] // self.num_agents)
                    self.split_channel = int(obs_shape[key].shape[0] // self.num_agents)
                    cnn_dims = np.array(obs_shape[key].shape[1:3], dtype=np.float32)
                else:
                    pass
        else:
            for key in self.cnn_keys:
                if key in ['rgb','depth','image','occupy_image']:
                    self.n_cnn_input += obs_shape[key].shape[2] 
                    cnn_dims = np.array(obs_shape[key].shape[:2], dtype=np.float32)
                elif key in ['global_map','local_map','global_obs','global_merge_obs', 'vector_cnn', 'global_merge_goal','gt_map']:
                    self.n_cnn_input += obs_shape[key].shape[0] 
                    cnn_dims = np.array(obs_shape[key].shape[1:3], dtype=np.float32)
                else:
                    raise NotImplementedError

        cnn_layers = []
        self.prev_out_channels = None
        for i, (out_channels, kernel_size, stride, padding) in enumerate(cnn_layers_params):
            if self._use_maxpool2d and i != len(cnn_layers_params) - 1:
                cnn_layers.append(nn.MaxPool2d(2))

            if i == 0:
                in_channels = self.n_cnn_input
            else:
                in_channels = self.prev_out_channels

            cnn_layers.append(init_(nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,)))
            #if i != len(cnn_layers_params) - 1:
            cnn_layers.append(active_func)
            self.prev_out_channels = out_channels

        for i, (_, kernel_size, stride, padding) in enumerate(cnn_layers_params):
            if self._use_maxpool2d and i != len(cnn_layers_params) - 1:
                cnn_dims = self._maxpool_output_dim(dimension=cnn_dims,
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array([2, 2], dtype=np.float32),
                stride=np.array([2, 2], dtype=np.float32))
            cnn_dims = self._cnn_output_dim(
                dimension=cnn_dims,
                padding=np.array([padding, padding], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array([kernel_size, kernel_size], dtype=np.float32),
                stride=np.array([stride, stride], dtype=np.float32),
            )
        
        to_linear = [
                Flatten(),
                init_(nn.Linear(cnn_layers_params[-1][0] * cnn_dims[0] * cnn_dims[1],
                            hidden_size)),
                active_func,
                nn.LayerNorm(hidden_size),
            ]
        if self.grid_goal:
            self.cnn_to_linear = nn.Sequential(*to_linear)
            self.cnn_to_grid = []
            for i, (out_channels, kernel_size, stride, padding) in enumerate(cnn_trans_layer):
                if i == 0:
                    in_channels = cnn_layers_params[-1][0]
                else:
                    in_channels = prev_out_channels
                self.cnn_to_grid.append(init_(nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,)))
                #if i != len(cnn_layers_params) - 1:
                self.cnn_to_grid.append(active_func)
                prev_out_channels = out_channels

                cnn_dims = self._cnn_output_dim(
                    dimension=cnn_dims,
                    padding=np.array([padding, padding], dtype=np.float32),
                    dilation=np.array([1, 1], dtype=np.float32),
                    kernel_size=np.array([kernel_size, kernel_size], dtype=np.float32),
                    stride=np.array([stride, stride], dtype=np.float32),
                )
            assert cnn_dims[0] == self.goal_grid_size and cnn_dims[1] == self.goal_grid_size
            self.cnn_to_grid = nn.Sequential(*self.cnn_to_grid)
        else:
            self.cnn_to_linear = nn.Sequential(*to_linear)
        return nn.Sequential(*cnn_layers)

    def _build_embed_model(self, obs_shape):
        self.embed_dim = 0
        for key in self.embed_keys:
            self.n_embed_input = 72
            self.n_embed_output = 8
            self.embed_dim += np.prod(obs_shape[key].shape)

        return nn.Embedding(self.n_embed_input, self.n_embed_output)

    def _build_mlp_model(self, obs_shape, hidden_size, use_orthogonal, activation_id):

        active_func = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(), nn.ELU()][activation_id]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu', 'leaky_relu', 'leaky_relu'][activation_id])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        for key in self.mlp_keys:
            self.n_mlp_input += np.prod(obs_shape[key].shape)

        return nn.Sequential(
                    init_(nn.Linear(self.n_mlp_input, hidden_size)), active_func, nn.LayerNorm(hidden_size))

    def _build_invariant_model(self):
        in_channels = self.prev_out_channels
        layers = [AlterEncoder(self.goal_grid_size ** 2, in_channels, depth=self.depth, hidden_dim = 64, heads = 4, dim_head = 16, mlp_dim = 64)]
        layers += [
                nn.Linear(64, 16),
                Rearrange('b (h w) c -> b c h w', h = self.goal_grid_size),
            ]
        return nn.Sequential(*layers)

    def _maxpool_output_dim(self, dimension, dilation, kernel_size, stride):
        """Calculates the output height and width based on the input
        height and width to the convolution layer.
        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(np.floor(
                    ((dimension[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]) + 1
                ))
            )
        return tuple(out_dimension)

    def _cnn_output_dim(self, dimension, padding, dilation, kernel_size, stride):
        """Calculates the output height and width based on the input
        height and width to the convolution layer.
        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(np.floor(
                    ((dimension[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]) + 1
                ))
            )
        return tuple(out_dimension)

    def _build_cnn_input(self, obs):
        cnn_input = []
        if self.use_stack:
            for key in self.cnn_keys:
                if key in ['stack_obs']:
                    cnn_input.append(obs[key])
                else:
                    pass
        else:
            for key in self.cnn_keys:
                if key in ['rgb','depth','image','occupy_image']:
                    cnn_input.append(obs[key].permute(0, 3, 1, 2) / 255.0)
                elif key in ['global_map', 'local_map', 'global_obs', 'global_merge_obs', 'vector_cnn', 'global_merge_goal','gt_map']:
                    cnn_input.append(obs[key])
                else:
                    raise NotImplementedError

        cnn_input = torch.cat(cnn_input, dim=1) 
        return cnn_input

    def _build_embed_input(self, obs):
        embed_input = []
        for key in self.embed_keys:
            embed_input.append(obs[key].view(obs[key].size(0), -1))

        embed_input = torch.cat(embed_input, dim=1)
        return embed_input

    def _build_mlp_input(self, obs):
        mlp_input = []
        for key in self.mlp_keys:
            mlp_input.append(obs[key].view(obs[key].size(0), -1))

        mlp_input = torch.cat(mlp_input, dim=1)
        return mlp_input

    @property
    def output_size(self):
        output_size = 0
        if len(self.cnn_keys) > 0:
            output_size += self.hidden_size

        if len(self.embed_keys) > 0:
            output_size += 8 * self.embed_dim

        if len(self.mlp_keys) > 0:
            output_size += self.hidden_size
        return output_size
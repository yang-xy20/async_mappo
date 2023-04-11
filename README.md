# Asynchronous Multi-Agent Reinforcement Learning for Efficient Real-Time Multi-Robot Cooperative Exploration

This is a PyTorch implementation of the paper: [Asynchronous Multi-Agent Reinforcement Learning for Efficient Real-Time Multi-Robot Cooperative Exploration](https://arxiv.org/abs/2301.03398)

Project Website: https://sites.google.com/view/ace-aamas

## Training

You could start training with by running `sh train_gridworld.sh` in directory [onpolicy/scripts](onpolicy/scripts). 

## Evaluation

Similar to training, you could run `sh render_gridworld.sh` in directory [onpolicy/scripts](onpolicy/scripts) to start evaluation. Remember to set up your path to the cooresponding model, correct hyperparameters and related evaluation parameters. 

We also provide our implementations of planning-based baselines. You could run `sh render_gridworld_ft.sh` to evaluate the planning-based methods. Note that `algorithm_name` determines the method to make global planning. It can be set to one of `mappo`, `ft_rrt`, `ft_apf`, `ft_nearest` and `ft_utility`.

You could also visualize the result and generate gifs by adding `--use_render` and `--save_gifs` to the scripts.

## Citation
If you find this repository useful, please cite our [paper](https://arxiv.org/abs/2301.03398):
```
@misc{yu2023asynchronous,
      title={Asynchronous Multi-Agent Reinforcement Learning for Efficient Real-Time Multi-Robot Cooperative Exploration}, 
      author={Chao Yu and Xinyi Yang and Jiaxuan Gao and Jiayu Chen and Yunfei Li and Jijia Liu and Yunfei Xiang and Ruixin Huang and Huazhong Yang and Yi Wu and Yu Wang},
      year={2023},
      eprint={2301.03398},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
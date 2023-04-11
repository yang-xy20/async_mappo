#!/bin/sh
env="GridWorld"
scenario="MiniGrid-MultiExploration-v0"
num_agents=1
grid_size=25
num_obstacles=0
local_step_num=1
seed_max=3
algo='ft_rrt'

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    echo "seed is ${seed}"
    exp=new_async_${algo}_grid${grid_size}_stepgoal_${local_step_num}_merge_normal
    CUDA_VISIBLE_DEVICES=3 python render/render_gridworld_ft.py\
      --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
      --num_agents ${num_agents} --num_obstacles ${num_obstacles} \
      --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --render_episodes 100 \
      --cnn_layers_params '16,3,1,1 32,3,1,1 16,3,1,1' \
      --ifi 0.5 --max_steps 300 --grid_size ${grid_size} --local_step_num ${local_step_num} --use_random_pos \
      --agent_view_size 7 --use_merge --use_merge_plan --use_eval \
      --astar_cost_mode "normal" --wandb_name "mapping" --user_name "yang-xy20" --asynch
done


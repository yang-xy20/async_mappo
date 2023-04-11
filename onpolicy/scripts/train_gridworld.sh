#!/bin/sh
env="GridWorld"
scenario="MiniGrid-MultiExploration-v0"
num_agents=3
num_obstacles=0
algo="mappo"
exp="async_global_new_attn_para_disc_single_overlap_normal50"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python train/train_gridworld.py \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
    --log_interval 1 --wandb_name "mapping" --user_name "yang-xy20" --num_agents ${num_agents} \
    --num_obstacles ${num_obstacles} --cnn_layers_params '16,7,2,1 32,5,2,1 16,3,1,1' --hidden_size 64 --seed 1 --n_training_threads 1 \
    --n_rollout_threads 50 --num_mini_batch 1 --num_env_steps 80000000 --ppo_epoch 3 --gain 0.01 \
    --lr 5e-4 --critic_lr 5e-4 --max_steps 150 --use_complete_reward --agent_view_size 7 --local_step_num 5 --use_random_pos \
    --astar_cost_mode normal  --cnn_trans_layer 1,3,1,1 --grid_size 25 --use_recurrent_policy \
    --use_global_goal --use_overlap_penalty --use_stack --goal_grid_size 5 --use_discrect --asynch
done
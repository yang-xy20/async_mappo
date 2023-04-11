#!/bin/sh
env="GridWorld"
scenario="MiniGrid-MultiExploration-v0"
num_agents=2
num_obstacles=0
algo="mappo"
exp="async_global_new_attn_para_single_to_async(1-5)"
seed_max=3

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=2 python render/render_gridworld.py\
      --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} \
      --num_agents ${num_agents} --num_obstacles ${num_obstacles} \
      --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --render_episodes 100 \
      --cnn_layers_params '16,7,2,1 32,5,2,1 16,3,1,1' \
      --model_dir "./results/GridWorld/MiniGrid-MultiExploration-v0/mappo/async_global_new_attn_para_no_agent_id_overlap_penalty_single_normal50/wandb/run-20220224_140614-3vhpd2ge/files/" \
      --max_steps 200 --use_complete_reward --agent_view_size 7 --local_step_num 1 --use_random_pos \
      --astar_cost_mode utility --grid_goal --goal_grid_size 5 --cnn_trans_layer 1,3,1,1 \
      --use_stack --grid_size 25 --use_recurrent_policy --use_stack --use_global_goal --use_overlap_penalty --use_eval --wandb_name "mapping" --user_name "yang-xy20" --asynch & 
done

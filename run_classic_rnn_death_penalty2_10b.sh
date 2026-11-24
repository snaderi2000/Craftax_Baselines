#!/usr/bin/env bash
set -euo pipefail
cd /home/shawheen/Craftax_Baselines
export CUDA_VISIBLE_DEVICES="${RNN_DEATH_PENALTY_GPU:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export WANDB_INIT_TIMEOUT=300 WANDB_MODE=online
rnn_death_penalty="${RNN_DEATH_PENALTY:-2.0}"
rnn_penalty_tag="${rnn_death_penalty%.0}"
rnn_penalty_tag="${rnn_penalty_tag//./p}"
rnn_diamond_reward="${RNN_DIAMOND_REWARD:-1.0}"
rnn_diamond_tag="${rnn_diamond_reward%.0}"
rnn_diamond_tag="${rnn_diamond_tag//./p}"
rnn_reward_suffix=""
if [[ "$rnn_diamond_tag" != "1" ]]; then
  rnn_reward_suffix="-diamond${rnn_diamond_tag}"
fi
exec /home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python -u ppo_rnn_og.py \
  --env_name Craftax-Classic-Symbolic-v1 --num_envs 1024 \
  --total_timesteps 1e10 --num_steps 64 --update_epochs 4 --num_minibatches 8 \
  --gamma 0.99 --gae_lambda 0.8 --clip_eps 0.2 --vf_coef 0.5 --ent_coef 0.01 \
  --lr 0.0002 --max_grad_norm 1.0 --anneal_lr --seed 50 --layer_size 512 \
  --death_penalty "$rnn_death_penalty" \
  --diamond_reward "$rnn_diamond_reward" \
  --run_name "${RNN_DEATH_PENALTY_RUN_NAME:-classic-rnn-death-penalty${rnn_penalty_tag}${rnn_reward_suffix}-10b-seed50}" \
  --wandb_project Craftax_Baselines --save_policy \
  --save_policy_milestones 65536,100000,150000,220000,330000,500000,750000,1000000,1500000,2200000,3300000,5000000,7500000,10000000,15000000,22000000,33000000,50000000,75000000,100000000,150000000,220000000,330000000,500000000,750000000,1000000000,1500000000,2200000000,3300000000,5000000000,7500000000

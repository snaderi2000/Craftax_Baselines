#!/usr/bin/env bash
set -euo pipefail

cd /home/shawheen/Craftax_Baselines

python /home/shawheen/Craftax_Baselines/ppo_rnd.py \
  --env_name Craftax-Classic-Symbolic-v1 \
  --num_envs 1024 \
  --total_timesteps 1e10 \
  --num_steps 64 \
  --update_epochs 4 \
  --num_minibatches 8 \
  --gamma 0.99 \
  --gae_lambda 0.8 \
  --clip_eps 0.2 \
  --vf_coef 0.5 \
  --ent_coef 0.01 \
  --lr 0.0002 \
  --max_grad_norm 1.0 \
  --anneal_lr \
  --seed 50 \
  --save_policy \
  --save_policy_milestones 65536,100000,150000,220000,330000,500000,750000,1000000,1500000,2200000,3300000,5000000,7500000,10000000,15000000,22000000,33000000,50000000,75000000,100000000,150000000,220000000,330000000,500000000,750000000,1000000000,1500000000,2200000000,3300000000,5000000000,7500000000

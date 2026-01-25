#!/bin/bash

# Define the range of seeds
for SEED in {5..10}
do
    echo "--------------------------------------------------"
    echo "Starting training for SEED: $SEED"
    echo "--------------------------------------------------"

    python ppo_rnn.py \
        --num_envs 48 \
        --total_timesteps 1e6 \
        --num_steps 96 \
        --update_epochs 4 \
        --num_minibatches 8 \
        --gamma 0.925 \
        --gae_lambda 0.625 \
        --clip_eps 0.2 \
        --vf_coef 1.0 \
        --ent_coef 0.01 \
        --lr 0.00045 \
        --max_grad_norm 0.5 \
        --anneal_lr \
        --no-use_optimistic_resets \
        --layer_size 2048 \
        --seed $SEED \
        --env_name Craftax-Classic-Pixels-v1 \
        --use_wandb  # Assuming you want to track all seeds separately

done
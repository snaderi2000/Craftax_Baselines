import d3rlpy
import h5py
import numpy as np
from tqdm import tqdm
import wandb
import torch
import dataclasses
from d3rlpy.algos import DiscreteCQLConfig
# ===================================================================
# 1. EXPERIMENT CONFIGURATION
# ===================================================================
WANDB_PROJECT = "Craftax-OfflineRL"
WANDB_RUN_NAME = "cql-run-1M-dataset"
EXPERIMENT_NAME = "DiscreteCQL_Craftax"



# Where to save the final trained model
MODEL_SAVE_PATH = "latest_20480_episodes_long.h5"

# Training hyperparameters
TOTAL_STEPS = 750_000
EPOCH_STEPS = 10_000  # How often to log, evaluate, and save checkpoints
BATCH_SIZE = 256     # Increased from the Atari example for better stability

# ===================================================================
# 2. DATA LOADING & PREPARATION
# ===================================================================
#print(f"Loading episodes from '{DATASET_PATH}'...")

with open("latest_2048_episodes.h5", "rb") as f:
    replay_buffer = d3rlpy.dataset.ReplayBuffer.load(
        f,
        d3rlpy.dataset.InfiniteBuffer()  # or FIFOBuffer if you want a limit
    )

print(f"Replay buffer loaded with {replay_buffer.transition_count} transitions.")



# ===================================================================
# 3. CQL ALGORITHM CONFIGURATION
# ===================================================================
print("Configuring Discrete CQL agent...")


cql = DiscreteCQLConfig().create(device="cuda:0")


# ===================================================================
# 4. SETTING UP EVALUATION AND TRAINING
# ===================================================================

# Since we can't run a live environment, we use offline evaluation metrics
# TD Error: Measures how well the Q-function is fitting the Bellman equation
wandb_logger = d3rlpy.logging.WanDBAdapterFactory(
    project=WANDB_PROJECT
)

td_error_evaluator = d3rlpy.metrics.TDErrorEvaluator()

# Initial State Value: Estimates the policy's performance on the dataset's starting states
initial_state_value_evaluator = d3rlpy.metrics.InitialStateValueEstimationEvaluator()


print("🚀 Starting training...")

cql.fit(
    replay_buffer,
    n_steps=TOTAL_STEPS,
    n_steps_per_epoch=EPOCH_STEPS,
    evaluators={
        "td_error": td_error_evaluator,
        "initial_state_value": initial_state_value_evaluator,
    },
    experiment_name=WANDB_RUN_NAME,
    with_timestamp=True,
    show_progress=True,
    logger_adapter=wandb_logger
)


# ===================================================================
# 5. SAVE FINAL MODEL
# ===================================================================
cql.save(MODEL_SAVE_PATH)
print(f"\n🎉 Training complete! Final model saved to {MODEL_SAVE_PATH}")


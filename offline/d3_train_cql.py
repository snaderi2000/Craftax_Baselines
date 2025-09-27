import d3rlpy
import h5py
import numpy as np
from tqdm import tqdm
import wandb
import torch
import dataclasses
# ===================================================================
# 1. EXPERIMENT CONFIGURATION
# ===================================================================
WANDB_PROJECT = "Craftax-OfflineRL"
WANDB_RUN_NAME = "cql-run-1M-dataset"
EXPERIMENT_NAME = "DiscreteCQL_Craftax"

# Path to the master dataset you created
DATASET_PATH = "episodes_compressed.h5"

# Where to save the final trained model
MODEL_SAVE_PATH = "cql_craftax_final.d3"

# Training hyperparameters
TOTAL_STEPS = 20_000
EPOCH_STEPS = 5_000  # How often to log, evaluate, and save checkpoints
BATCH_SIZE = 256     # Increased from the Atari example for better stability

# ===================================================================
# 2. DATA LOADING & PREPARATION
# ===================================================================
print(f"Loading episodes from '{DATASET_PATH}'...")

# Create a list of d3rlpy.dataset.Episode objects from our custom HDF5 file
episodes = []
with h5py.File(DATASET_PATH, "r") as hf:
    for episode_key in tqdm(hf.keys(), desc="Loading episodes"):
        episode_group = hf[episode_key]
        
        actions = episode_group['action'][:].reshape(-1, 1)
        rewards = episode_group['reward'][:].reshape(-1, 1)
        is_terminated = episode_group['done'][-1]

        episode = d3rlpy.dataset.Episode( 
            observations=episode_group['obs'][:],
            actions=actions,
            rewards=rewards,
            terminated=is_terminated
        )
        episodes.append(episode)

# The ReplayBuffer holds all data and handles sampling
replay_buffer = d3rlpy.dataset.ReplayBuffer(
    buffer=d3rlpy.dataset.FIFOBuffer(limit=len(episodes) * 1024), # Set a large limit
    episodes=episodes
)

print(f"\n✅ Replay Buffer is ready with {replay_buffer.transition_count} transitions.")


# ===================================================================
# 3. CQL ALGORITHM CONFIGURATION
# ===================================================================
print("Configuring Discrete CQL agent...")

# Configure the Discrete CQL algorithm, borrowing good hyperparameters from the Atari example
# cql = d3rlpy.algos.DiscreteCQLConfig(
#     batch_size=BATCH_SIZE,
#     learning_rate=5e-5,
#     optim_factory=d3rlpy.optimizers.AdamFactory(eps=1e-2 / BATCH_SIZE),
    
#     # The conservative penalty weight. This is the most important CQL hyperparameter.
#     alpha=4.0,
    
#     # Quantile Regression is a powerful Q-function often used in discrete offline RL
#     q_func_factory=d3rlpy.models.q_functions.QRQFunctionFactory(n_quantiles=200),
    
#     # We don't use a PixelScaler because our observations are vectors, not images
#     observation_scaler=None,
    
#     # Standard practice to clip rewards for stability
#     reward_scaler=d3rlpy.preprocessing.ClipRewardScaler(-1.0, 1.0),
    
#     target_update_interval=2000,

# ).create(device='cuda:0' if torch.cuda.is_available() else 'cpu')

# Configure the Discrete CQL algorithm, borrowing good hyperparameters from the Atari example
cql_config = d3rlpy.algos.DiscreteCQLConfig(
    batch_size=BATCH_SIZE,
    learning_rate=5e-5,
    optim_factory=d3rlpy.optimizers.AdamFactory(eps=1e-2 / BATCH_SIZE),
    alpha=4.0,
    q_func_factory=d3rlpy.models.q_functions.QRQFunctionFactory(n_quantiles=200),
    observation_scaler=None,
    reward_scaler=d3rlpy.preprocessing.ClipRewardScaler(-1.0, 1.0),
    target_update_interval=2000,
)
cql = cql_config.create(device='cuda:0' if torch.cuda.is_available() else 'cpu')

# ===================================================================
# 4. SETTING UP EVALUATION AND TRAINING
# ===================================================================

# Since we can't run a live environment, we use offline evaluation metrics
# TD Error: Measures how well the Q-function is fitting the Bellman equation
wandb_logger = d3rlpy.logging.WanDBAdapterFactory(
    project=WANDB_PROJECT,  
    config=dataclasses.asdict(cql_config) # Pass the config object here
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


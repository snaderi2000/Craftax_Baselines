import os
import numpy as np
import wandb
from d3rlpy.dataset import ReplayBuffer, FIFOBuffer, Transition
from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.dataset.transition_pickers import TransitionPickerProtocol

# ===============================
# CONFIGURATION
# ===============================
WANDB_PROJECT = "Craftax-OfflineRL"
WANDB_RUN_NAME = "cql_run_1"
EXPERIMENT_NAME = "craftax_cql"

DATASET_PATH = "craftax_dataset_cleaned.h5"   # Preprocessed dataset
MODEL_SAVE_PATH = "cql_craftax_final.d3"      # Final saved model
TOTAL_STEPS = 500_000
EPOCH_STEPS = 10_000                          # How often logs/checkpoints are written
BATCH_SIZE = 32

# ===============================
# FIX: Custom Transition Picker
# ===============================
class DiscreteTransitionPicker(TransitionPickerProtocol):
    """
    Ensures actions and rewards are always returned as 2D arrays
    with shapes compatible for batching in d3rlpy.
    """

    def __call__(self, episode, index: int) -> Transition:
        obs = episode.observations[index]

        # Determine terminal
        is_terminal = episode.terminated and index == episode.size() - 1
        if is_terminal:
            next_obs = np.zeros_like(obs)
        else:
            next_obs = episode.observations[index + 1]

        # Wrap scalar values as 2D
        action = np.array([episode.actions[index]], dtype=np.int32)    # (1,)
        reward = np.array([episode.rewards[index]], dtype=np.float32)  # (1,)

        return Transition(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            terminal=float(is_terminal),
            interval=1
        )

# ===============================
# MAIN SCRIPT
# ===============================
def main():
    # ---- W&B Setup ----
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)
    print(f"Using W&B project: {WANDB_PROJECT}, run name: {WANDB_RUN_NAME}")

    # ---- Load Dataset ----
    buffer = FIFOBuffer(limit=None)
    with open(DATASET_PATH, "rb") as f:
        replay_buffer = ReplayBuffer.load(
            f,
            buffer,
            transition_picker=DiscreteTransitionPicker()
        )

    # Dataset info
    info = replay_buffer.dataset_info
    print(f"\nDataset loaded:")
    print(f"  Episodes: {len(replay_buffer.episodes)}")
    print(f"  Total transitions: {replay_buffer.transition_count}")
    print(f"  Action space: {info.action_space}")
    print(f"  Action size: {info.action_size}")
    print(f"  Observation signature: {info.observation_signature}")

    # ---- Verify Transition Batch ----
    batch = replay_buffer.sample_transition_batch(batch_size=BATCH_SIZE)
    print("\nSampled batch check:")
    print(f"  Actions shape: {batch.actions.shape} | dtype: {batch.actions.dtype}")
    print(f"  Rewards shape: {batch.rewards.shape} | dtype: {batch.rewards.dtype}")
    print(f"  First 5 actions: {batch.actions[:5].flatten()}")
    print(f"  First 5 rewards: {batch.rewards[:5].flatten()}")

    # ---- Initialize CQL ----
    cql = DiscreteCQLConfig().create(device="cuda:0")
    print("\nCQL agent initialized on GPU.")

    # ---- Train ----
    print("\nStarting training...")
    cql.fit(
        replay_buffer,
        n_steps=TOTAL_STEPS,
        n_steps_per_epoch=EPOCH_STEPS,
        experiment_name=EXPERIMENT_NAME,
        with_timestamp=True,
        show_progress=True
    )

    # ---- Save Final Model ----
    cql.save(MODEL_SAVE_PATH)
    print(f"\nTraining complete! Model saved to: {MODEL_SAVE_PATH}")

    # ---- Finish W&B ----
    wandb.finish()

if __name__ == "__main__":
    main()

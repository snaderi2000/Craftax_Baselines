import os
import numpy as np
import wandb
from d3rlpy.dataset import ReplayBuffer, FIFOBuffer, Transition
from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy.dataset.transition_pickers import TransitionPickerProtocol
from d3rlpy.logging import WanDBAdapterFactory

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
    """Fix for d3rlpy requiring extra fields: next_action and rewards_to_go."""
    def __call__(self, episode, index: int) -> Transition:
        obs = episode.observations[index]

        # Check if this is the final step
        is_terminal = episode.terminated and index == episode.size() - 1

        # Next observation
        next_obs = np.zeros_like(obs) if is_terminal else episode.observations[index + 1]

        # ---- FIX 1: Wrap action and reward in arrays ----
        action = np.array([episode.actions[index]], dtype=np.int32)       # shape (1,)
        reward = np.array([episode.rewards[index]], dtype=np.float32)     # shape (1,)

        # ---- FIX 2: Add next_action ----
        if is_terminal:
            next_action = np.zeros_like(action)
        else:
            next_action = np.array([episode.actions[index + 1]], dtype=np.int32)

        # ---- FIX 3: Compute rewards_to_go ----
        rewards_to_go = np.array(
            [np.sum(episode.rewards[index:])], dtype=np.float32
        )  # shape (1,)

        return Transition(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=next_obs,
            terminal=float(is_terminal),
            interval=1,
            next_action=next_action,
            rewards_to_go=rewards_to_go,
        )

# ===============================
# MAIN SCRIPT
# ===============================
def main():
    # ---- Initialize W&B ----
    wandb_logger = WanDBAdapterFactory(project=WANDB_PROJECT)

    print(f"Using W&B project: {WANDB_PROJECT})

    # ---- Load Dataset ----
    buffer = FIFOBuffer(limit=None)
    with open(DATASET_PATH, "rb") as f:
        replay_buffer = ReplayBuffer.load(
            f,
            buffer,
            transition_picker=DiscreteTransitionPicker()  # custom picker fix
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
    print("  Next actions shape:", batch.next_actions.shape)
    print("  Rewards-to-go shape:", batch.rewards_to_go.shape)

    # ---- Initialize CQL ----
    cql = DiscreteCQLConfig().create(device="cuda:0")
    print("\nCQL agent initialized on GPU.")

    cql.build_with_dataset(replay_buffer)

    wandb_logger = WanDBAdapterFactory(project=WANDB_PROJECT)

    print("Checking W&B Logger Factory...")
    wandb_logger_instance = wandb_logger.create(cql, "craftax_cql", EPOCH_STEPS)
    print("Logger created:", wandb_logger_instance)



    # ---- Train ----
    print("\nStarting training...")
    cql.fit(
        replay_buffer,
        n_steps=TOTAL_STEPS,
        n_steps_per_epoch=EPOCH_STEPS,
        experiment_name=EXPERIMENT_NAME,
        logger_adapter=wandb_logger,  # <- Integrates directly with W&B
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

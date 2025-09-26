import wandb
from d3rlpy.dataset import ReplayBuffer, FIFOBuffer
from d3rlpy.algos import DiscreteCQLConfig
from d3rlpy import load_learnable
import os

# ===============================
# CONFIGURATION
# ===============================
WANDB_PROJECT = "Craftax-OfflineRL"
WANDB_RUN_NAME = "cql_run_1"
EXPERIMENT_NAME = "craftax_cql"
DATASET_PATH = "craftax_dataset_5files_phases.h5"
MODEL_SAVE_PATH = "cql_craftax_final.d3"
TOTAL_STEPS = 500_000
EPOCH_STEPS = 10_000   # how often logs and checkpoints are written

# ===============================
# MAIN SCRIPT
# ===============================
def main():
    # ---- W&B Setup ----
    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)
    
    # ---- Load Dataset ----
    buffer = FIFOBuffer(limit=None)
    with open(DATASET_PATH, "rb") as f:
        replay_buffer = ReplayBuffer.load(f, buffer)

    print(f"Dataset loaded: {len(replay_buffer.episodes)} episodes, "
          f"{replay_buffer.transition_count} transitions")
    print(f"Action space: {replay_buffer.action_space}, "
          f"Action size: {replay_buffer.action_size}")

    # ---- Initialize CQL ----
    cql = DiscreteCQLConfig().create(device="cuda:0")
    print("CQL agent initialized on GPU.")

    # ---- Train ----
    cql.fit(
        replay_buffer,
        n_steps=TOTAL_STEPS,
        n_steps_per_epoch=EPOCH_STEPS,  # evaluation + logs every X steps
        experiment_name=EXPERIMENT_NAME,
        with_timestamp=True,            # separate logs for each run
        show_progress=True
    )

    # ---- Save Final Model ----
    cql.save(MODEL_SAVE_PATH)
    print(f"Training complete! Model saved to {MODEL_SAVE_PATH}")

    wandb.finish()

if __name__ == "__main__":
    main()

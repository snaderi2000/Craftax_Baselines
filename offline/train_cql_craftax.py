import wandb
from d3rlpy.algos import CQL
from d3rlpy.dataset import ReplayBuffer, FIFOBuffer
import os

# ===============================
# CONFIGURATION
# ===============================
DATASET_PATH = "craftax_dataset_cleaned.h5"
EXPERIMENT_NAME = "craftax_cql"
WANDB_PROJECT = "Craftax-OfflineRL"
WANDB_ENTITY = None  # Optional: set to your W&B username or team
TOTAL_STEPS = 500_000
MODEL_SAVE_PATH = "cql_craftax_final_model.pt"

# ===============================
# MAIN TRAINING SCRIPT
# ===============================
def main():
    # --- Initialize W&B ---
    wandb.init(project=WANDB_PROJECT, name=EXPERIMENT_NAME)

    # --- Load Offline Dataset ---
    print(f"Loading dataset from {DATASET_PATH} ...")
    buffer = FIFOBuffer(limit=None)
    with open(DATASET_PATH, "rb") as f:
        replay_buffer = ReplayBuffer.load(f, buffer)

    print(f"Dataset loaded: {len(replay_buffer.episodes)} episodes, {replay_buffer.transition_count} transitions")

    # --- Initialize CQL Agent ---
    print("Initializing CQL agent ...")
    cql = CQL(use_gpu=True)  # Automatically uses cuda:0

    # --- Train Agent ---
    print("Starting training ...")
    cql.fit(
        replay_buffer,
        n_steps=TOTAL_STEPS,
        experiment_name=EXPERIMENT_NAME,  # Directory for logs
        with_timestamp=True,              # Adds timestamp to folder name
        show_progress=True                # Progress bar during training
    )

    # --- Save Final Model ---
    print(f"Saving final model to {MODEL_SAVE_PATH} ...")
    cql.save_model(MODEL_SAVE_PATH)

    # --- Finish W&B run ---
    wandb.finish()
    print("Training complete!")

if __name__ == "__main__":
    main()

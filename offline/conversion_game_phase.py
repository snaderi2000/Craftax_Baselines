import os
import random
import numpy as np
import h5py
from tqdm import tqdm

# -----------------------
# CONFIGURATION
# -----------------------
DATA_DIR = "/home/synaderi/Craftax_Baselines/craftax_classic_200M_dataset"  # folder with .npz files
OUTPUT_PATH = "craftax_dataset_5files_phases.h5"  # output HDF5 file
NUM_FILES_TO_SAMPLE = 5  # number of .npz files to sample
FINAL_TRANSITIONS_LIMIT = 4_000_000  # cap total transitions to avoid memory issues
SEED = 42
# -----------------------

random.seed(SEED)
np.random.seed(SEED)


def compute_threshold_per_file(steps):
    """
    Compute the 95th percentile threshold for one file.
    """
    return np.percentile(steps, 95)


def assign_phase(step, threshold):
    """
    Assign a phase based on normalized step count.
    Early = 0, Mid = 1, Late = 2
    """
    if step < 0.3 * threshold:
        return 0
    elif step < 0.7 * threshold:
        return 1
    else:
        return 2


def convert_to_h5(files, output_path, final_limit=None):
    """
    Convert selected .npz files into a single .h5 file with per-file phase labeling.
    """
    obs_list, actions_list, rewards_list = [], [], []
    next_obs_list, terminals_list, phases_list = [], [], []
    episode_ids_list, steps_list = [], []

    total_transitions = 0

    for path in tqdm(files, desc="Processing files"):
        data = np.load(path, allow_pickle=True)

        obs = data["obs"]
        next_obs = data["next_obs"]
        actions = data["action"]
        rewards = data["reward"]
        dones = data["done"]
        steps = data["step_in_episode"]
        episode_ids = data["episode_id"]

        # --- Step 1: compute per-file threshold ---
        threshold = compute_threshold_per_file(steps)

        # --- Step 2: assign phases relative to this file ---
        phases = np.array([assign_phase(s, threshold) for s in steps], dtype=np.int32)

        # --- Step 3: prepare terminal flags ---
        terminals = dones.astype(np.bool_)

        # Append data
        obs_list.append(obs)
        next_obs_list.append(next_obs)
        actions_list.append(actions)
        rewards_list.append(rewards)
        terminals_list.append(terminals)
        phases_list.append(phases)
        episode_ids_list.append(episode_ids)
        steps_list.append(steps)


        total_transitions += len(obs)
        if final_limit and total_transitions >= final_limit:
            print(f"Reached limit of {final_limit} transitions. Stopping early.")
            break

    # Concatenate everything
    obs_array = np.concatenate(obs_list)[:final_limit]
    next_obs_array = np.concatenate(next_obs_list)[:final_limit]
    actions_array = np.concatenate(actions_list)[:final_limit]
    rewards_array = np.concatenate(rewards_list)[:final_limit]
    terminals_array = np.concatenate(terminals_list)[:final_limit]
    phases_array = np.concatenate(phases_list)[:final_limit]
    episode_ids_array = np.concatenate(episode_ids_list)[:final_limit]
    steps_array = np.concatenate(steps_list)[:final_limit]

    print("\nFinal dataset shapes:")
    print("Observations:", obs_array.shape)
    print("Next Observations:", next_obs_array.shape)
    print("Actions:", actions_array.shape)
    print("Rewards:", rewards_array.shape)
    print("Terminals:", terminals_array.shape)
    print("Phases:", phases_array.shape)
    print("Episode IDs:", episode_ids_array.shape)
    print("Steps in Episode:", steps_array.shape)

    # Save to HDF5
    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("observations", data=obs_array, compression="gzip")
        hf.create_dataset("next_observations", data=next_obs_array, compression="gzip")
        hf.create_dataset("actions", data=actions_array, compression="gzip")
        hf.create_dataset("rewards", data=rewards_array, compression="gzip")
        hf.create_dataset("terminals", data=terminals_array, compression="gzip")
        hf.create_dataset("phases", data=phases_array, compression="gzip")
        hf.create_dataset("episode_ids", data=episode_ids_array, compression="gzip")
        hf.create_dataset("steps_in_episode", data=steps_array, compression="gzip")

    print(f"\n✅ Saved merged dataset with per-file phases to: {output_path}")


if __name__ == "__main__":
    # Step 1: Select random subset of .npz files
    all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".npz")]
    if len(all_files) == 0:
        raise RuntimeError(f"No .npz files found in {DATA_DIR}")

    sampled_files = random.sample(all_files, NUM_FILES_TO_SAMPLE)
    print(f"Randomly selected {len(sampled_files)} files for processing: {sampled_files}")

    # Step 2: Convert to HDF5
    convert_to_h5(sampled_files, OUTPUT_PATH, final_limit=FINAL_TRANSITIONS_LIMIT)

import os
import random
import numpy as np
from tqdm import tqdm
import d3rlpy

# -----------------------
# CONFIGURATION
# -----------------------
DATA_DIR = "/home/synaderi/Craftax_Baselines/craftax_classic_200M_dataset"  # folder with .npz files
OUTPUT_PATH = "craftax_dataset_flattened.h5"  # output file
NUM_FILES_TO_SAMPLE = None  # set to None to use all files, or an integer like 64
SEED = 42
# -----------------------

def convert_npz_to_mdpdataset(data_dir, output_path, num_files=None, seed=42):
    # Collect all .npz files
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]
    if len(all_files) == 0:
        raise RuntimeError(f"No .npz files found in {data_dir}")
    print(f"Found {len(all_files)} total files in {data_dir}")

    # Optionally sample a subset of files
    if num_files is not None:
        random.seed(seed)
        sampled_files = random.sample(all_files, min(num_files, len(all_files)))
        print(f"Randomly selected {len(sampled_files)} files")
    else:
        sampled_files = all_files
        print("Using all files.")

    # Storage for merged transitions
    all_obs = []
    all_actions = []
    all_rewards = []
    all_dones = []

    # Process each file
    for path in tqdm(sampled_files, desc="Loading files"):
        data = np.load(path, allow_pickle=True)

        # Extract only the fields we care about
        all_obs.append(data["obs"])
        all_actions.append(data["action"])
        all_rewards.append(data["reward"])
        all_dones.append(data["done"].astype(np.float32))  # ensure float32 for compatibility

    # Concatenate into one large array
    observations = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)
    terminals = np.concatenate(all_dones, axis=0)

    print(f"Final merged size: {observations.shape[0]} transitions")
    print(f"Observation shape: {observations.shape[1:]}")

    # Create MDPDataset
    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    # Save to HDF5
    with open(output_path, "w+b") as f:
        dataset.dump(f)

    print(f"Saved MDPDataset to {output_path}")


if __name__ == "__main__":
    convert_npz_to_mdpdataset(DATA_DIR, OUTPUT_PATH, NUM_FILES_TO_SAMPLE, SEED)

import os
import random
import numpy as np
from tqdm import tqdm
from d3rlpy.dataset import MDPDataset

# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = "craftax_classic_200M_dataset"  # folder with .npz files
OUTPUT_PATH = "craftax_dataset_10percent.h5"
SAMPLE_FRACTION = 0.10        # 10% of all files
SEED = 42                      # reproducibility
# ----------------------------

def convert_random_subset_to_h5(data_dir, output_path, fraction=0.10, seed=42):
    random.seed(seed)

    # 1. Collect all .npz files
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]
    total_files = len(files)
    if total_files == 0:
        raise RuntimeError(f"No .npz files found in {data_dir}")
    print(f"Found {total_files} files in {data_dir}")

    # 2. Randomly sample fraction
    sample_size = max(1, int(total_files * fraction))
    sampled_files = random.sample(files, sample_size)
    print(f"Randomly selected {sample_size} files ({fraction*100:.1f}% of total)")

    # 3. Load data from sampled files
    observations, actions, rewards, dones, next_observations = [], [], [], [], []

    for file_path in tqdm(sampled_files, desc="Converting"):
        data = np.load(file_path)

        # extract fields
        obs = data["obs"]
        next_obs = data["next_obs"]
        act = data["action"]
        rew = data["reward"]
        done = data["done"]

        observations.append(obs)
        next_observations.append(next_obs)
        actions.append(act)
        rewards.append(rew)
        dones.append(done)

    # 4. Stack into arrays
    observations = np.concatenate(observations, axis=0).astype(np.float32)
    next_observations = np.concatenate(next_observations, axis=0).astype(np.float32)
    actions = np.concatenate(actions, axis=0).astype(np.int32)
    rewards = np.concatenate(rewards, axis=0).astype(np.float32)
    dones = np.concatenate(dones, axis=0).astype(np.float32)

    print(f"Final dataset shape: {observations.shape[0]} transitions")

    # 5. Create and save MDPDataset
    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=dones
    )
    dataset.dump(output_path)
    print(f"Saved MDPDataset to {output_path}")


if __name__ == "__main__":
    convert_random_subset_to_h5(DATA_DIR, OUTPUT_PATH, SAMPLE_FRACTION, SEED)

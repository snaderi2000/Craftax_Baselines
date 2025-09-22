import os
import random
import numpy as np
from tqdm import tqdm
from d3rlpy.dataset import MDPDataset

# === Settings ===
DATA_DIR = "craftax_classic_200M_dataset"   # Directory with .npz files
OUTPUT_PATH = "craftax_dataset_10percent.h5"
SAMPLE_FRACTION = 0.10                       # 10% of files
RANDOM_SEED = 42                             # For reproducibility

# === Helper ===
def convert_npz_to_mdp(data_dir, output_path, sample_fraction=0.10, seed=42):
    random.seed(seed)

    # List all .npz files
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]
    total_files = len(all_files)
    print(f"Found {total_files} .npz files total.")

    # Select random subset
    subset_size = max(1, int(total_files * sample_fraction))
    chosen_files = random.sample(all_files, subset_size)
    print(f"Randomly selected {subset_size} files ({sample_fraction*100:.0f}%).")

    obs_list, act_list, rew_list, terminal_list = [], [], [], []

    # Process each file
    for file_path in tqdm(chosen_files, desc="Converting"):
        data = np.load(file_path)

        obs = data["obs"]          # shape (N, obs_dim)
        actions = data["action"]   # shape (N,)
        rewards = data["reward"]   # shape (N,)
        dones = data["done"]       # shape (N,)

        # Convert boolean dones to float32 terminal flags
        terminals = dones.astype(np.float32)

        obs_list.append(obs)
        act_list.append(actions)
        rew_list.append(rewards)
        terminal_list.append(terminals)

    # Stack everything into single arrays
    observations = np.concatenate(obs_list, axis=0)
    actions = np.concatenate(act_list, axis=0)
    rewards = np.concatenate(rew_list, axis=0)
    terminals = np.concatenate(terminal_list, axis=0)

    print(f"Final dataset shape: {observations.shape[0]:,} transitions")
    print(f"Obs dim: {observations.shape[1]}, Action dim: {actions.shape[-1] if actions.ndim > 1 else 1}")

    # Save as .h5
    dataset = MDPDataset(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         terminals=terminals)
    dataset.dump(output_path)
    print(f"Saved subset to {output_path}")

# === Main ===
if __name__ == "__main__":
    convert_npz_to_mdp(DATA_DIR, OUTPUT_PATH, sample_fraction=SAMPLE_FRACTION, seed=RANDOM_SEED)

# convert_to_mdp.py
import os
import numpy as np
from d3rlpy.dataset import MDPDataset
from tqdm import tqdm

DATA_DIR = "/home/synaderi/Craftax_Baselines/craftax_classic_200M_dataset"
OUTPUT_PATH = "craftax_dataset_5percent.h5"

def convert_npz_to_mdp(data_dir, output_path, subset_fraction=0.05):
    obs_list, next_obs_list = [], []
    actions, rewards, dones = [], [], []

    npz_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])

    print(f"Found {len(npz_files)} files. Using {subset_fraction*100:.1f}% of each file.")

    for file in tqdm(npz_files, desc="Converting"):
        path = os.path.join(data_dir, file)
        data = np.load(path, allow_pickle=False)

        N = data["obs"].shape[0]

        # Randomly sample 5% of the transitions from this file
        keep_count = int(N * subset_fraction)
        indices = np.random.choice(N, keep_count, replace=False)

        obs_list.append(data["obs"][indices])
        next_obs_list.append(data["next_obs"][indices])
        actions.append(data["action"][indices])
        rewards.append(data["reward"][indices])
        dones.append(data["done"][indices].astype(np.float32))

    # Concatenate into single arrays
    observations = np.concatenate(obs_list, axis=0)
    next_observations = np.concatenate(next_obs_list, axis=0)
    actions = np.concatenate(actions, axis=0)
    rewards = np.concatenate(rewards, axis=0)
    dones = np.concatenate(dones, axis=0)

    print(f"Subset size: {observations.shape[0]} transitions")

    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=dones,
        next_observations=next_observations,
    )

    dataset.dump(output_path)
    print(f"Saved subset MDPDataset to {output_path}")


if __name__ == "__main__":
    convert_npz_to_mdp(DATA_DIR, OUTPUT_PATH, subset_fraction=0.05)

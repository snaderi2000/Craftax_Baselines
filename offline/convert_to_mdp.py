# convert_to_mdp.py
import os
import numpy as np
from d3rlpy.dataset import MDPDataset
from tqdm import tqdm

DATA_DIR = "/home/synaderi/Craftax_Baselines/craftax_classic_200M_dataset"
OUTPUT_PATH = "craftax_dataset_5percent.h5"

def convert_npz_to_mdp(data_dir, output_path, max_files=50):
    obs_list, next_obs_list = [], []
    actions, rewards, dones = [], [], []

    # Grab only the first `max_files` files
    npz_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npz")])[:max_files]

    print(f"Found {len(npz_files)} files to convert...")

    for file in tqdm(npz_files, desc="Converting"):
        path = os.path.join(data_dir, file)
        data = np.load(path, allow_pickle=False)

        obs_list.append(data["obs"])
        next_obs_list.append(data["next_obs"])
        actions.append(data["action"])
        rewards.append(data["reward"])
        dones.append(data["done"].astype(np.float32))

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
    convert_npz_to_mdp(DATA_DIR, OUTPUT_PATH, max_files=50)

import h5py
import numpy as np
from tqdm import tqdm
import math
import random

# --- Configuration ---
SOURCE_H5_PATH = "episodes_subset_50_files.h5"
TARGET_H5_PATH = "final_weighted_trajectory_dataset.h5"

# The final dataset will contain ALL trajectories from the last 33% of each episode's steps.
LATE_PHASE_PERCENTILE = 33
# ---

# 1. LOAD AND PARTITION EPISODES INTO TRAJECTORIES
print(f"Loading and partitioning episodes from '{SOURCE_H5_PATH}'...")
early_phase_trajectories = []
late_phase_trajectories = []

with h5py.File(SOURCE_H5_PATH, "r") as hf:
    for episode_key in tqdm(hf.keys(), desc="Partitioning Episodes"):
        episode = {key: hf[episode_key][key][:] for key in hf[episode_key].keys()}
        num_steps = len(episode['obs'])
        split_index = int(num_steps * (1 - LATE_PHASE_PERCENTILE / 100.0))
        
        # Create early-phase trajectory (a shorter, terminated episode)
        early_traj = {key: data[:split_index] for key, data in episode.items()}
        if len(early_traj['obs']) > 0:
            # Mark this truncated trajectory as "done" at its new end
            early_traj['done'][-1] = True
            early_phase_trajectories.append(early_traj)
            
        # Create late-phase trajectory
        late_traj = {key: data[split_index:] for key, data in episode.items()}
        if len(late_traj['obs']) > 0:
            late_phase_trajectories.append(late_traj)

# 2. PREPARE THE FINAL SET OF TRAJECTORIES
# The late-phase data is ALL the trajectories from that pool.
final_trajectories = late_phase_trajectories
num_late_transitions = sum(len(e['obs']) for e in final_trajectories)

# Determine how many early-phase transitions to add to meet the ~66%/34% split
total_transitions_target = int(num_late_transitions / 0.66)
num_early_transitions_needed = total_transitions_target - num_late_transitions

# Randomly sample from the early-phase trajectories until we have enough transitions
print(f"\nIncluding all {len(late_phase_trajectories)} late-phase trajectories ({num_late_transitions} transitions).")
print(f"Sampling early-phase trajectories to gather ~{num_early_transitions_needed} transitions...")

early_phase_samples = []
current_early_transitions = 0
random.shuffle(early_phase_trajectories) # Shuffle to get a random sample
for traj in early_phase_trajectories:
    if current_early_transitions >= num_early_transitions_needed:
        break
    early_phase_samples.append(traj)
    current_early_transitions += len(traj['obs'])

final_trajectories.extend(early_phase_samples)
random.shuffle(final_trajectories) # Shuffle the final combined list of trajectories

# 3. SAVE THE FINAL DATASET
print(f"\nSaving {len(final_trajectories)} final trajectories to '{TARGET_H5_PATH}'...")
keys_to_compress = {'obs', 'next_obs'}

with h5py.File(TARGET_H5_PATH, 'w') as hf:
    for i, trajectory in enumerate(tqdm(final_trajectories, desc="Saving trajectories")):
        episode_group = hf.create_group(f"episode_{i}")
        for key, data in trajectory.items():
            if key in keys_to_compress:
                episode_group.create_dataset(
                    key, data=data, compression="gzip", compression_opts=4
                )
            else:
                episode_group.create_dataset(key, data=data)

print(f"\n✅ Success! Your final dataset is ready at '{TARGET_H5_PATH}'.")

import h5py
import numpy as np

# Open the HDF5 file
with h5py.File("craftax_dataset_5files_phases.h5", "r") as f:
    phases = f["phases"][:]           # game phase labels
    steps = f["steps_in_episode"][:]   # step counts
    episode_ids = f["episode_ids"][:] # episode IDs for clarity

    # Randomly sample 10 indices
    num_samples = 10
    random_indices = np.random.choice(len(phases), size=num_samples, replace=False)

    print("Random Sample of 10 Transitions:\n")
    for idx in random_indices:
        phase_label = {0: "Early", 1: "Mid", 2: "Late"}[phases[idx]]
        print(f"Index {idx:6d} | "
              f"Episode ID: {episode_ids[idx]:8d} | "
              f"Step in Episode: {steps[idx]:4d} | "
              f"Phase: {phase_label} ({phases[idx]})")


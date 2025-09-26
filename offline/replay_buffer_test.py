import h5py
import numpy as np
from d3rlpy.dataset import ReplayBuffer, FIFOBuffer, Signature, Transition
from d3rlpy.constants import ActionSpace
import os

# ====================================================
# CONFIGURATION
# ====================================================
FILE_PATH = "/home/synaderi/Craftax_Baselines/offline/craftax_dataset_5files_phases.h5"
SAVE_DIR = "/home/synaderi/Craftax_Baselines/offline"
SAVE_PATH = os.path.join(SAVE_DIR, "craftax_replay_buffer.d3")

# ====================================================
# LOAD HDF5 DATASET
# ====================================================
print("Loading dataset from:", FILE_PATH)
with h5py.File(FILE_PATH, "r") as f:
    observations = f["observations"][:]
    next_observations = f["next_observations"][:]
    actions = f["actions"][:]
    rewards = f["rewards"][:]
    terminals = f["terminals"][:].astype(np.float32)

# Verify counts
N = len(observations)
assert N == len(next_observations) == len(actions) == len(rewards) == len(terminals)
print(f"✅ Loaded dataset with {N:,} transitions.")

# Determine action space size
unique_actions = np.unique(actions)
action_size = len(unique_actions)
print("Unique actions:", unique_actions)
print("Action size:", action_size)

# ====================================================
# INITIALIZE REPLAY BUFFER
# ====================================================
# Use string dtypes, and (1,) for scalar signatures
observation_signature = Signature(shape=(1345,), dtype="float32")
action_signature = Signature(shape=(1,), dtype="int32")      # discrete scalar
reward_signature = Signature(shape=(1,), dtype="float32")    # scalar reward

buffer = FIFOBuffer(limit=None)
replay_buffer = ReplayBuffer(
    buffer=buffer,
    observation_signature=observation_signature,
    action_signature=action_signature,
    reward_signature=reward_signature,
    action_size=action_size,
    action_space=ActionSpace.DISCRETE
)
print("ReplayBuffer initialized successfully!")

# ====================================================
# APPEND TRANSITIONS
# ====================================================
print("Building ReplayBuffer...")

for i in range(N):
    # Wrap action and reward as shape (1,) arrays
    action = np.array([actions[i]], dtype=np.int32)
    reward = np.array([rewards[i]], dtype=np.float32)

    # Append using positional arguments ONLY
    replay_buffer.append(
        observations[i],
        action,
        reward,
        next_observations[i],
        terminals[i],
    )

    if (i + 1) % 50000 == 0:
        print(f"  Processed {i + 1:,} / {N:,} transitions")

print(f"✅ ReplayBuffer built successfully with {replay_buffer.transition_count:,} transitions.")



# ====================================================
# SAVE REPLAY BUFFER
# ====================================================
print("Saving ReplayBuffer to:", SAVE_PATH)
with open(SAVE_PATH, "wb") as f:
    replay_buffer.dump(f)

print("🎉 ReplayBuffer saved successfully!")

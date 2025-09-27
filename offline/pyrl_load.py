import h5py
import torch
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage

# ---- Load HDF5 data ----
with h5py.File("craftax_dataset_cleaned.h5", "r") as f:
    observations = torch.tensor(f["observations"][:], dtype=torch.float32)
    next_observations = torch.tensor(f["next_observations"][:], dtype=torch.float32)
    actions = torch.tensor(f["actions"][:], dtype=torch.int64)
    rewards = torch.tensor(f["rewards"][:], dtype=torch.float32)
    terminals = torch.tensor(f["terminals"][:], dtype=torch.bool)
    phases = torch.tensor(f["phases"][:], dtype=torch.int64)

# ---- Build transition dict ----
transitions = {
    "observation": observations,
    "action": actions,
    "reward": rewards,
    "done": terminals,
    "next_observation": next_observations,
    "phase": phases,  # extra metadata
}

# Convert to TensorDict
from tensordict import TensorDict
tensordict = TensorDict(transitions, batch_size=[observations.shape[0]])

# ---- Replay buffer ----
storage = LazyTensorStorage(max_size=observations.shape[0])
rb = TensorDictReplayBuffer(storage=storage)
rb.extend(tensordict)

print("Replay buffer size:", len(rb))


sample = rb.sample(5)  # sample 5 transitions
print("Sampled transitions:")
print("Observations:", sample["observation"].shape)
print("Actions:", sample["action"])
print("Rewards:", sample["reward"])
print("Phases:", sample["phase"])
print("Done flags:", sample["done"])


unique, counts = torch.unique(phases, return_counts=True)
for u, c in zip(unique.tolist(), counts.tolist()):
    label = {0: "Early", 1: "Mid", 2: "Late"}[u]
    print(f"{label} ({u}): {c} transitions, {c/len(phases)*100:.2f}%")



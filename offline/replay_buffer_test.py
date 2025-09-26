import h5py
import numpy as np
from d3rlpy.dataset import ReplayBuffer, FIFOBuffer, Signature, TransitionPickerProtocol, Transition
from d3rlpy.constants import ActionSpace

# ========================================
# 1. Load Dataset
# ========================================
file_path = "/home/synaderi/Craftax_Baselines/offline/craftax_dataset_5files_phases.h5"
with h5py.File(file_path, "r") as f:
    observations = f["observations"][:]
    next_observations = f["next_observations"][:]
    actions = f["actions"][:]
    rewards = f["rewards"][:]
    terminals = f["terminals"][:].astype(np.float32)

print(f"✅ Loaded dataset with {len(observations):,} transitions")

# ========================================
# 2. Custom Transition Picker
# ========================================
class CustomNextObsTransitionPicker(TransitionPickerProtocol):
    def __init__(self, next_obs_array, terminals_array):
        self.next_obs_array = next_obs_array
        self.terminals_array = terminals_array

    def __call__(self, episode, index: int) -> Transition:
        return Transition(
            observation=episode.observations[index],
            action=episode.actions[index],
            reward=episode.rewards[index],
            next_observation=self.next_obs_array[index],  # ✅ use correct s'
            terminal=float(self.terminals_array[index]),
            interval=1
        )

# ========================================
# 3. Initialize ReplayBuffer WITH Picker
# ========================================
buffer = FIFOBuffer(limit=None)
obs_sig = Signature(shape=(1345,), dtype="float32")
act_sig = Signature(shape=(1,), dtype="int32")
rew_sig = Signature(shape=(1,), dtype="float32")

transition_picker = CustomNextObsTransitionPicker(next_observations, terminals)

replay_buffer = ReplayBuffer(
    buffer=buffer,
    transition_picker=transition_picker,  # ✅ pass here directly
    observation_signature=obs_sig,
    action_signature=act_sig,
    reward_signature=rew_sig,
    action_size=len(np.unique(actions)),
    action_space=ActionSpace.DISCRETE
)

print("ReplayBuffer initialized with custom transition picker!")

# ========================================
# 4. Sanity Check
# ========================================
batch = replay_buffer.sample_transition_batch(batch_size=4)
print("Observation batch shape:", batch.observations.shape)
print("Next observation batch shape:", batch.next_observations.shape)

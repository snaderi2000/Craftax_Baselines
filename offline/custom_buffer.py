import d3rlpy
import numpy as np
import os
from dataclasses import dataclass, asdict
from typing import List

# ===============================================================
# PART 1: DEFINITIONS - Our Custom Components
# ===============================================================
print("--- Part 1: Defining Custom Components ---")

class CustomEpisode(d3rlpy.dataset.Episode):
    def __init__(self, observations, actions, rewards, terminals, progress):
        is_episode_terminated = bool(terminals[-1])
        super().__init__(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminated=is_episode_terminated,
        )
        object.__setattr__(self, 'terminals', np.asarray(terminals, dtype=np.float32))
        object.__setattr__(self, 'progress', np.asarray(progress, dtype=np.float32))

@dataclass(frozen=True)
class ProgressTransition(d3rlpy.dataset.Transition):
    progress: float

@dataclass(frozen=True)
class ProgressTransitionMiniBatch(d3rlpy.dataset.TransitionMiniBatch):
    rewards_to_go: np.ndarray
    progress: np.ndarray

    @classmethod
    def from_progress_transitions(cls, transitions: List[ProgressTransition]) -> "ProgressTransitionMiniBatch":
        observations = np.array([t.observation for t in transitions])
        actions = np.array([t.action for t in transitions])
        rewards = np.array([t.reward for t in transitions], dtype=np.float32).reshape(-1, 1)
        next_observations = np.array([t.next_observation for t in transitions])
        terminals = np.array([t.terminal for t in transitions], dtype=np.float32).reshape(-1, 1)
        next_actions = np.array([t.next_action for t in transitions])
        rewards_to_go = np.vstack([t.rewards_to_go for t in transitions]).astype(np.float32)
        progress = np.array([t.progress for t in transitions], dtype=np.float32).reshape(-1, 1)
        
        # ✅ FIX: Change dtype of intervals to float32 to match library's internal validation.
        intervals = np.array([t.interval for t in transitions], dtype=np.float32).reshape(-1, 1)
        
        return cls(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminals=terminals,
            next_actions=next_actions,
            rewards_to_go=rewards_to_go,
            progress=progress,
            intervals=intervals,
            transitions=transitions
        )

class ProgressTransitionPicker(d3rlpy.dataset.BasicTransitionPicker):
    def __call__(self, episode: CustomEpisode, index: int) -> ProgressTransition:
        parent_transition = super().__call__(episode, index)
        transition_dict = asdict(parent_transition)
        transition_dict['progress'] = episode.progress[index]
        return ProgressTransition(**transition_dict)

class ProgressReplayBuffer(d3rlpy.dataset.ReplayBuffer):
    def _build_progress_distribution(self):
        all_progress = [ep.progress for ep in self.episodes]
        progress_array = np.concatenate(all_progress).ravel() + 1e-6
        return progress_array / np.sum(progress_array)

    def sample_transition_batch(self, batch_size: int) -> ProgressTransitionMiniBatch:
        probabilities = self._build_progress_distribution()
        sampled_indices = np.random.choice(self.transition_count, size=batch_size, p=probabilities)
        
        transitions = []
        for index in sampled_indices:
            episode, transition_index = self._buffer[index]
            transition = self._transition_picker(episode, transition_index)
            transitions.append(transition)
            
        return ProgressTransitionMiniBatch.from_progress_transitions(transitions)
        
print("✅ Custom Components Defined.\n")

# ===============================================================
# PART 2: DATA LOADING & PREPARATION
# ===============================================================
print("--- Part 2: Loading and Preparing Data ---")

file_name = "latest_2048_episodes.h5"
if not os.path.exists(file_name):
    raise FileNotFoundError(f"Data file '{file_name}' not found!")

with open(file_name, "rb") as f:
    buffer = d3rlpy.dataset.InfiniteBuffer()
    loaded_dataset = d3rlpy.dataset.ReplayBuffer.load(f, buffer=buffer)

new_episodes_with_progress = []
for episode in loaded_dataset.episodes:
    progress_data = np.linspace(0.0, 1.0, num=episode.size(), dtype=np.float32)
    terminals_data = np.zeros(episode.size(), dtype=np.float32)
    if episode.terminated:
        terminals_data[-1] = 1.0
    new_episodes_with_progress.append(
        CustomEpisode(
            episode.observations,
            episode.actions,
            episode.rewards,
            terminals_data,
            progress_data,
        )
    )
print(f"✅ Converted {len(new_episodes_with_progress)} episodes.\n")

# ===============================================================
# PART 3: USING THE CUSTOM BUFFER & TESTING
# ===============================================================
print("--- Part 3: Initializing and Testing the ProgressReplayBuffer ---")

progress_buffer = ProgressReplayBuffer(
    buffer=d3rlpy.dataset.InfiniteBuffer(),
    episodes=new_episodes_with_progress,
    transition_picker=ProgressTransitionPicker()
)

print(f"✅ Custom buffer initialized with {progress_buffer.transition_count} transitions.\n")

# --- Verification Step ---
print("--- Verifying Sampling Distribution ---")
batch_size = 5000
batch = progress_buffer.sample_transition_batch(batch_size)
sampled_progress = batch.progress

print(f"Sampled {batch_size} transitions.")
print(f"Average progress in a uniform sample would be ~0.5")
print(f"Average progress in our weighted sample: {sampled_progress.mean():.4f}")

if sampled_progress.mean() > 0.55:
    print("✅ Success! The average progress is higher than 0.5, indicating successful prioritized sampling.")
else:
    print("⚠️ Warning: The average progress is not significantly higher than 0.5. Check the implementation.")

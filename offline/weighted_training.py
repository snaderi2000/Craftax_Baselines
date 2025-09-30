import d3rlpy
import numpy as np
import torch
import argparse
from dataclasses import dataclass, asdict
from typing import List
from d3rlpy.algos import DiscreteCQLConfig

# (All custom components like CustomEpisode, ProgressTransition, etc. are the same as before)
class CustomEpisode(d3rlpy.dataset.Episode):
    def __init__(self, observations, actions, rewards, terminals, progress):
        is_episode_terminated = bool(terminals[-1])
        super().__init__(
            observations=observations, actions=actions, rewards=rewards,
            terminated=is_episode_terminated)
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
        intervals = np.array([t.interval for t in transitions], dtype=np.float32).reshape(-1, 1)
        return cls(
            observations=observations, actions=actions, rewards=rewards,
            next_observations=next_observations, terminals=terminals,
            next_actions=next_actions, rewards_to_go=rewards_to_go,
            progress=progress, intervals=intervals, transitions=transitions)

class ProgressTransitionPicker(d3rlpy.dataset.BasicTransitionPicker):
    def __call__(self, episode: CustomEpisode, index: int) -> ProgressTransition:
        parent_transition = super().__call__(episode, index)
        transition_dict = asdict(parent_transition)
        transition_dict['progress'] = episode.progress[index]
        return ProgressTransition(**transition_dict)


class StratifiedProgressReplayBuffer(d3rlpy.dataset.ReplayBuffer):
    def __init__(self, buffer, episodes, threshold, high_progress_ratio, uniform_within_strata=False, **kwargs):
        super().__init__(buffer, episodes=episodes, **kwargs)
        self.threshold = threshold
        self.high_progress_ratio = high_progress_ratio
        self.uniform_within_strata = uniform_within_strata
        
        mode = "UNIFORM" if self.uniform_within_strata else "WEIGHTED"
        print(f"Stratified sampling enabled (mode: {mode}): threshold={threshold}, ratio={high_progress_ratio}")
        self._prepare_stratified_pools()

    def _prepare_stratified_pools(self):
        # ... (this method remains the same)
        print("Preparing stratified sampling pools...")
        all_progress = np.concatenate([ep.progress for ep in self.episodes]).ravel()
        all_indices = np.arange(self.transition_count)
        self.high_indices = all_indices[all_progress >= self.threshold]
        self.low_indices = all_indices[all_progress < self.threshold]
        print(f"High-progress pool size: {len(self.high_indices)}")
        print(f"Low-progress pool size: {len(self.low_indices)}")
        if not self.uniform_within_strata:
            high_vals = all_progress[self.high_indices] + 1e-6
            low_vals = all_progress[self.low_indices] + 1e-6 if len(self.low_indices) > 0 else []
            self.high_probs = high_vals / np.sum(high_vals)
            self.low_probs = low_vals / np.sum(low_vals) if len(low_vals) > 0 else []

    def sample_transition_batch(self, batch_size: int) -> ProgressTransitionMiniBatch:
        n_high = int(batch_size * self.high_progress_ratio)
        n_low = batch_size - n_high
        
        sampled_indices = []
        
        # ✅ FIX: Conditionally sample with or without weights
        if n_high > 0:
            if self.uniform_within_strata:
                high_samples = np.random.choice(self.high_indices, size=n_high, replace=True)
            else:
                high_samples = np.random.choice(self.high_indices, size=n_high, p=self.high_probs, replace=True)
            sampled_indices.append(high_samples)
            
        if n_low > 0:
            if self.uniform_within_strata:
                low_samples = np.random.choice(self.low_indices, size=n_low, replace=True)
            else:
                low_samples = np.random.choice(self.low_indices, size=n_low, p=self.low_probs, replace=True)
            sampled_indices.append(low_samples)
            
        combined_indices = np.concatenate(sampled_indices)
        np.random.shuffle(combined_indices)
        
        transitions = [self._picker_helper(index) for index in combined_indices]
        return ProgressTransitionMiniBatch.from_progress_transitions(transitions)

    def _picker_helper(self, index: int) -> ProgressTransition:
        episode, transition_index = self._buffer[index]
        return self._transition_picker(episode, transition_index)

def main(args):
    # ... (Setup logic is the same)
    wandb_run_name = args.wandb_run_name or f"cql-{args.strategy}"
    wandb_logger = d3rlpy.logging.WanDBAdapterFactory(project=args.wandb_project)
    
    print(f"Loading dataset from '{args.dataset}'...")
    with open(args.dataset, "rb") as f:
        buffer = d3rlpy.dataset.InfiniteBuffer()
        loaded_dataset = d3rlpy.dataset.ReplayBuffer.load(f, buffer=buffer)

    if args.strategy == 'uniform':
        print("Using standard ReplayBuffer for true UNIFORM sampling.")
        replay_buffer = loaded_dataset
    
    else: # 'weighted' strategy now has a sub-mode
        print("Using StratifiedProgressReplayBuffer.")
        # Data conversion is only needed for our custom buffer
        episodes_with_progress = []
        for episode in loaded_dataset.episodes:
            progress_data = np.linspace(0.0, 1.0, num=episode.size(), dtype=np.float32)
            terminals_data = np.zeros(episode.size(), dtype=np.float32)
            if episode.terminated: terminals_data[-1] = 1.0
            episodes_with_progress.append(
                CustomEpisode(
                    episode.observations, episode.actions, episode.rewards,
                    terminals_data, progress_data,
                )
            )
        
        replay_buffer = StratifiedProgressReplayBuffer(
            buffer=d3rlpy.dataset.InfiniteBuffer(),
            episodes=episodes_with_progress,
            transition_picker=ProgressTransitionPicker(),
            threshold=args.threshold,
            high_progress_ratio=args.ratio,
            uniform_within_strata=args.uniform_within_strata # Pass the new flag
        )
    
    print(f"✅ Buffer ready with {replay_buffer.transition_count} transitions.\n")
    
    # --- CQL ALGORITHM CONFIGURATION & TRAINING (remains the same) ---
    print("Configuring Discrete CQL agent...")
    BATCH_SIZE = 256
    

    cql = DiscreteCQLConfig().create(device="cuda:0") 

    print("🚀 Starting training...")
    cql.fit(
        replay_buffer, n_steps=500_000, n_steps_per_epoch=10_000,
        evaluators={
            "td_error": d3rlpy.metrics.TDErrorEvaluator(),
            "initial_state_value": d3rlpy.metrics.InitialStateValueEstimationEvaluator(),
        },
        experiment_name=wandb_run_name, with_timestamp=True,
        show_progress=True, logger_adapter=wandb_logger
    )

    cql.save(args.output)
    print(f"\n🎉 Training complete! Final model saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CQL agent with stratified sampling.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the .h5 dataset file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the final trained model.")
    parser.add_argument("--strategy", type=str, required=True, choices=["weighted", "uniform"], help="Use 'uniform' for control, 'weighted' for stratified sampling.")
    parser.add_argument("--threshold", type=float, default=0.6, help="For 'weighted' strategy: the progress value to split high/low pools.")
    parser.add_argument("--ratio", type=float, default=0.7, help="For 'weighted' strategy: the ratio of samples from the high-progress pool.")
    parser.add_argument("--uniform_within_strata", action='store_true', help="For 'weighted' strategy: if set, sample uniformly within strata instead of by progress.")
    parser.add_argument("--wandb_project", type=str, default="Craftax-OfflineRL-Stratified", help="W&B project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name.")
    args = parser.parse_args()
    main(args)

import numpy as np
from d3rlpy.dataset import ReplayBuffer, FIFOBuffer, Episode, Transition, BasicTransitionPicker

# =========================================================
# Custom Transition Picker
# =========================================================
class DiscreteTransitionPicker(BasicTransitionPicker):
    def __call__(self, episode, index):
        # Use default BasicTransitionPicker logic
        t = super().__call__(episode, index)

        # Wrap action and reward to ensure correct shape
        wrapped_action = np.array([t.action], dtype=np.int32)      # shape (1,)
        wrapped_reward = np.array([t.reward], dtype=np.float32)    # shape (1,)

        return Transition(
            observation=t.observation,
            action=wrapped_action,                       # FIXED
            reward=wrapped_reward,                       # FIXED
            next_observation=t.next_observation,
            terminal=t.terminal,
            interval=t.interval,
            next_action=np.array([0], dtype=np.int32),   # placeholder
            rewards_to_go=np.array([0.0], dtype=np.float32)  # placeholder
        )


# =========================================================
# Create Toy Dataset
# =========================================================
def create_toy_episodes():
    """
    Create two toy episodes with 5 steps each.
    Observations have dim=3, actions are scalar discrete values.
    """
    obs = np.random.randn(5, 3).astype(np.float32)
    rewards = np.random.randn(5).astype(np.float32)

    # Episode 1
    actions1 = np.array([0, 1, 2, 1, 0], dtype=np.int32)
    episode1 = Episode(observations=obs, actions=actions1, rewards=rewards, terminated=True)

    # Episode 2
    actions2 = np.array([1, 0, 2, 2, 1], dtype=np.int32)
    episode2 = Episode(observations=obs, actions=actions2, rewards=rewards, terminated=True)

    return [episode1, episode2]

# =========================================================
# Main Test
# =========================================================
def main():
    # Create episodes
    episodes = create_toy_episodes()

    # Initialize replay buffer with custom transition picker
    buffer = FIFOBuffer(limit=None)
    replay_buffer = ReplayBuffer(
        buffer=buffer,
        transition_picker=DiscreteTransitionPicker(),
        episodes=episodes
    )

    print(f"ReplayBuffer initialized with {len(replay_buffer.episodes)} episodes")

    # Sample a single transition
    t = replay_buffer.sample_transition()
    print("Single transition:")
    print("  Observation shape:", t.observation.shape)
    print("  Action:", t.action, "Shape:", t.action.shape, "Dtype:", t.action.dtype)

    # Sample a batch of transitions
    batch = replay_buffer.sample_transition_batch(batch_size=4)
    print("\nSampled batch:")
    print("  Batch actions shape:", batch.actions.shape)
    print("  Batch actions:\n", batch.actions)

if __name__ == "__main__":
    main()

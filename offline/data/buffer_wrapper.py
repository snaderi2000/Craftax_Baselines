# offline/data/buffer_wrapper.py
import numpy as np
from tianshou.data import ReplayBuffer, Batch


class NPZReplayBuffer(ReplayBuffer):
    """
    A wrapper around Tianshou ReplayBuffer to push batches of data
    coming from .npz datasets (e.g., offline RL datasets).

    Args:
        buffer_size (int): Maximum size of the buffer.
        obs_dim (int): Dimension of the observation vector.
        action_dim (int): Number of possible discrete actions.
        kwargs: Other arguments passed to Tianshou ReplayBuffer.
    """
    def __init__(self, buffer_size, obs_dim, action_dim, **kwargs):
        super().__init__(size=buffer_size, **kwargs)
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def push_batch(self, obs, actions, rewards, dones, next_obs):
        """
        Push a batch of transitions into the buffer.

        Args:
            obs (np.ndarray): Shape [batch_size, obs_dim]
            actions (np.ndarray): Shape [batch_size]
            rewards (np.ndarray): Shape [batch_size]
            dones (np.ndarray): Boolean array, shape [batch_size]
            next_obs (np.ndarray): Shape [batch_size, obs_dim]
        """
        batch = Batch(
            obs=obs,
            act=actions,
            rew=rewards,
            terminated=dones.astype(np.bool_),  # Tianshou expects 'terminated'
            truncated=np.zeros_like(dones, dtype=np.bool_),  # No truncation for now
            obs_next=next_obs,
            info={}  # Empty info dict
        )
        self.add(batch)

# offline/data/buffer_wrapper.py
import numpy as np
from tianshou.data import ReplayBuffer, Batch

class NPZReplayBuffer(ReplayBuffer):
    """
    A wrapper for Tianshou ReplayBuffer to push batches of data
    from .npz files into the buffer one transition at a time.
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
            dones (np.ndarray): Shape [batch_size], booleans
            next_obs (np.ndarray): Shape [batch_size, obs_dim]
        """
        batch_size = obs.shape[0]
        for i in range(batch_size):
            transition = Batch(
                obs=obs[i],
                act=actions[i],
                rew=rewards[i],
                terminated=bool(dones[i]),
                truncated=False,
                obs_next=next_obs[i],
                info={}
            )
            self.add(transition)

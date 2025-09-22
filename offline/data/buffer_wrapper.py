import numpy as np
from tianshou.data import ReplayBuffer

class NPZReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, obs_dim, action_dim, **kwargs):
        super().__init__(buffer_size, **kwargs)
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def push_batch(self, obs, actions, rewards, dones, next_obs):
        """Push a batch of transitions into the replay buffer."""       
        data = Batch(
        obs=obs,
        act=actions,
        rew=rewards,
        done=dones,
        obs_next=next_obs,
        info={}  # now correctly nested inside Batch
    )
    self.add(data)  # pass Batch instead of individual keyword args


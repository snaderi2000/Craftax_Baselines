from tianshou.data import ReplayBuffer, Batch

class NPZReplayBuffer(ReplayBuffer):
    def __init__(self, size, **kwargs):
        super().__init__(size=size, **kwargs)

    def push_batch(self, obs, actions, rewards, dones, next_obs):
        """Push a batch of transitions into the replay buffer."""
        data = Batch(
            obs=obs,
            act=actions,
            rew=rewards,
            done=dones,
            obs_next=next_obs,
            info={}  # wrap info properly inside Batch
        )
        self.add(data)  # ✅ self is valid here because we are inside a method

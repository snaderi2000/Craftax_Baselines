"""Real-episode Monte Carlo replay for positive-advantage self-imitation."""
import numpy as np
from symbolic_ez_replay import SumTree


def discounted_episode_returns(rewards,gamma=.99):
    result=np.empty(len(rewards),np.float32);future=0.
    for t in range(len(rewards)-1,-1,-1):
        future=float(rewards[t])+gamma*future;result[t]=future
    return result


class EpisodeReplay:
    def __init__(self,num_envs,capacity=131072,seed=50):
        self.capacity=capacity;self.size=0;self.position=0;self.rng=np.random.default_rng(seed)
        self.obs=np.empty((capacity,1345),np.float16)
        self.actions=np.empty(capacity,np.int16);self.returns=np.empty(capacity,np.float32)
        self.tree=SumTree(capacity);self.max_priority=1.
        self.pending=[[] for _ in range(num_envs)];self.completed=0

    def add_rollout(self,obs,actions,rewards,dones):
        for t in range(len(rewards)):
            for env in range(rewards.shape[1]):
                self.pending[env].append((obs[t,env].astype(np.float16),int(actions[t,env]),float(rewards[t,env])))
                if dones[t,env]:
                    episode=self.pending[env];self.pending[env]=[];self.completed+=1
                    returns=discounted_episode_returns([r[2] for r in episode])
                    if len(episode)>self.capacity:episode=episode[-self.capacity:];returns=returns[-self.capacity:]
                    n=len(episode);ids=(self.position+np.arange(n))%self.capacity
                    self.obs[ids]=np.stack([r[0] for r in episode])
                    self.actions[ids]=[r[1] for r in episode];self.returns[ids]=returns
                    self.tree.set(ids,np.full(n,self.max_priority**.6))
                    self.position=(self.position+n)%self.capacity;self.size=min(self.capacity,self.size+n)

    def sample(self,n):
        if not self.size:raise ValueError('No completed episode available')
        ids=self.tree.sample(self.rng,n)
        ids=np.where(self.rng.random(n)<.2,self.rng.integers(self.size,size=n),ids)
        probs=.2/self.size+.8*self.tree.tree[self.tree.size+ids]/self.tree.tree[1]
        weights=probs**(-.4);weights=(weights/weights.max()).astype(np.float32)
        return dict(obs=self.obs[ids].astype(np.float32),action=self.actions[ids].astype(np.int32),
            returns=self.returns[ids].copy(),weights=weights),ids

    def update(self,ids,advantages):
        values=np.clip(np.asarray(advantages)+1e-3,1e-3,100.)
        unique,inverse=np.unique(ids,return_inverse=True);maximum=np.zeros(len(unique))
        np.maximum.at(maximum,inverse,values)
        self.tree.set(unique,maximum**.6);self.max_priority=max(self.max_priority,float(values.max()))

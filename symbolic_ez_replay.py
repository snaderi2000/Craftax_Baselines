# SPDX-License-Identifier: GPL-3.0-only
"""Compact, sequence-safe real-transition replay with mixed prioritized sampling."""
import numpy as np
from models.symbolic_ez import BLOCKS,CELLS,MAP_DIM


def pack_observation(obs):
    obs=np.asarray(obs)
    view=obs[...,:MAP_DIM].reshape(*obs.shape[:-1],CELLS,BLOCKS+4)
    block=view[...,:BLOCKS].argmax(-1).astype(np.uint8)
    mobs=np.sum((view[...,BLOCKS:]>.5).astype(np.uint8)*(1<<np.arange(4,dtype=np.uint8)),axis=-1).astype(np.uint8)
    return block,mobs,obs[...,MAP_DIM:].astype(np.float16)


def unpack_observation(block,mobs,stats):
    terrain=np.eye(BLOCKS,dtype=np.float32)[block]
    presence=((mobs[...,None]>>np.arange(4,dtype=np.uint8))&1).astype(np.float32)
    local=np.concatenate([terrain,presence],axis=-1).reshape(*stats.shape[:-1],MAP_DIM)
    return np.concatenate([local,stats.astype(np.float32)],axis=-1)


class SumTree:
    def __init__(self,n):
        self.size=1<<(int(n)-1).bit_length()
        self.tree=np.zeros(2*self.size,np.float64)

    def set(self,indices,values):
        indices=np.asarray(indices,dtype=np.int64)+self.size
        self.tree[indices]=values
        while indices.size and indices[0]>1:
            indices=np.unique(indices//2)
            self.tree[indices]=self.tree[2*indices]+self.tree[2*indices+1]

    def sample(self,rng,n):
        mass=rng.random(n)*self.tree[1];indices=np.ones(n,dtype=np.int64)
        while indices[0]<self.size:
            left=2*indices;go_right=mass>=self.tree[left]
            mass-=np.where(go_right,self.tree[left],0.)
            indices=left+go_right
        return indices-self.size


class SequenceReplay:
    def __init__(self,num_envs,capacity_steps=512,width=256,burnin=64,unroll=5,td=5,seed=50):
        self.n=num_envs;self.capacity=capacity_steps;self.width=width
        self.burnin=burnin;self.unroll=unroll;self.td=td;self.time=0
        self.future=unroll+td
        self.rng=np.random.default_rng(seed)
        shape=(capacity_steps,num_envs)
        self.data={}
        for prefix in ('','next_'):
            self.data[prefix+'block']=np.empty((*shape,CELLS),np.uint8)
            self.data[prefix+'mobs']=np.empty((*shape,CELLS),np.uint8)
            self.data[prefix+'stats']=np.empty((*shape,22),np.float16)
        for name,dtype in [('action',np.int16),('prev_action',np.int16),('reward',np.float32),
            ('prev_reward',np.float32),('done',np.float32),('reset',bool),('search_value',np.float32)]:
            self.data[name]=np.empty(shape,dtype)
        self.data['policy']=np.empty((*shape,17),np.float16)
        self.data['hidden']=np.empty((*shape,width),np.float16)
        self.tree=SumTree(capacity_steps*num_envs);self.max_priority=1.
        self.times=np.full(capacity_steps,-1,np.int64)

    @property
    def valid_range(self):
        return max(0,self.time-self.capacity)+self.burnin,self.time-self.future

    def append(self,obs,next_obs,**fields):
        t=self.time%self.capacity
        indices=t*self.n+np.arange(self.n)
        self.tree.set(indices,np.zeros(self.n))
        for prefix,observation in (('',obs),('next_',next_obs)):
            for name,a in zip(('block','mobs','stats'),pack_observation(observation)):
                self.data[prefix+name][t]=a
        for name,a in fields.items():self.data[name][t]=a
        self.times[t]=self.time;self.time+=1
        start=self.time-self.future-1
        lo,hi=self.valid_range
        if lo<=start<hi:
            self.tree.set((start%self.capacity)*self.n+np.arange(self.n),np.full(self.n,self.max_priority**.6))

    def sample(self,batch_size):
        lo,hi=self.valid_range
        if hi<=lo:raise ValueError('Not enough real history/future transitions for replay')
        anchors=[]
        while sum(len(x) for x in anchors)<batch_size:
            n=batch_size-sum(len(x) for x in anchors)
            slots=self.tree.sample(self.rng,n)
            uniform=self.rng.random(n)<.2
            ut=self.rng.integers(lo,hi,size=n);ue=self.rng.integers(self.n,size=n)
            slots=np.where(uniform,(ut%self.capacity)*self.n+ue,slots)
            absolute=self.times[slots//self.n]
            anchors.append(slots[(absolute>=lo)&(absolute<hi)])
        slots=np.concatenate(anchors)[:batch_size]
        start=self.times[slots//self.n];env=slots%self.n
        absolute=start[None,:]+np.arange(-self.burnin,self.future+1)[:,None]
        time=absolute%self.capacity
        assert np.array_equal(self.times[time],absolute),'Replay sequence overwritten'
        batch={k:v[time,env[None,:]].copy() for k,v in self.data.items()
               if k not in ('hidden','block','mobs','stats','next_block','next_mobs','next_stats')}
        for prefix in ('','next_'):
            batch[prefix+'obs']=unpack_observation(*[self.data[prefix+k][time,env[None,:]] for k in ('block','mobs','stats')])
        batch['initial_h']=self.data['hidden'][time[0],env].astype(np.float32)
        batch['policy']=batch['policy'].astype(np.float32)
        batch['policy']/=batch['policy'].sum(-1,keepdims=True)
        leaves=self.tree.tree[self.tree.size:self.tree.size+self.capacity*self.n].reshape(self.capacity,self.n)
        valid=(self.times>=lo)&(self.times<hi)
        valid_mass=leaves[valid].sum()
        prob=.2/((hi-lo)*self.n)+.8*self.tree.tree[self.tree.size+slots]/valid_mass
        weights=prob**(-.4);batch['weights']=(weights/weights.max()).astype(np.float32)
        # Internal indices are only for replay bookkeeping, never model inputs.
        return batch,slots

    def update_priorities(self,slots,priorities):
        priorities=np.clip(np.asarray(priorities),1e-3,100.)
        # Duplicate sampled anchors must have a deterministic aggregate.
        unique,inverse=np.unique(slots,return_inverse=True)
        maximum=np.zeros(len(unique));np.maximum.at(maximum,inverse,priorities)
        self.tree.set(unique,maximum**.6)
        self.max_priority=max(self.max_priority,float(priorities.max()))

    def reanalyse(self,slots,policy,values):
        for slot,p,v in zip(slots,np.asarray(policy),np.asarray(values)):
            self.data['policy'][slot//self.n,slot%self.n]=p
            self.data['search_value'][slot//self.n,slot%self.n]=v

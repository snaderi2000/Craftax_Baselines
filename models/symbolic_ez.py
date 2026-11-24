# Symbolic JAX adaptation, 2026. SPDX-License-Identifier: GPL-3.0-only
# Training losses / residual dynamics adapted from EfficientZeroV2:
# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
# Reference commit 12b9e775df1f8afb4a1dc854cf722ea84fbfb61d.
# See third_party/EfficientZeroV2/LICENSE and SYMBOLIC_EZ_NOTICE.md.
"""History-conditioned symbolic dynamics, search, and multi-step losses."""
import sys,functools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/mctx_py'))
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import mctx
from craftax.craftax_classic.constants import BlockType,OBS_DIM

BLOCKS=len(BlockType)
CELLS=int(np.prod(OBS_DIM))
MAP_DIM=CELLS*(BLOCKS+4)
SUPPORT=jnp.linspace(-5.,5.,201)


def symlog(x):return jnp.sign(x)*jnp.log1p(jnp.abs(x))
def symexp(x):return jnp.sign(x)*jnp.expm1(jnp.abs(x))


def scalar_target(x):
    y=jnp.clip(symlog(x),SUPPORT[0],SUPPORT[-1])
    index=(y+5.)*((len(SUPPORT)-1)/10.)
    low=jnp.floor(index).astype(jnp.int32);high=jnp.minimum(low+1,len(SUPPORT)-1)
    fraction=index-low
    return jax.nn.one_hot(low,len(SUPPORT))*(1-fraction[...,None])+jax.nn.one_hot(high,len(SUPPORT))*fraction[...,None]


def scalar_decode(logits):return symexp(jnp.sum(jax.nn.softmax(logits)*SUPPORT,axis=-1))
def cross_entropy(logits,target):return -(jax.nn.log_softmax(logits)*target).sum(-1)


def scale_gradient(x,scale):
    # Equivalent to the reference dynamics-state gradient hook, forward unchanged.
    return jax.lax.stop_gradient(x)*(1-scale)+x*scale


class SymbolicEZ(nn.Module):
    width:int=256
    actions:int=17

    def setup(self):
        self.obs_layer=nn.Dense(self.width)
        self.gru=nn.GRUCell(self.width)
        self.dyn_in=nn.Dense(self.width)
        self.dyn_out=nn.Dense(self.width)
        self.dyn_norm=nn.LayerNorm()
        self.pred_layer=nn.Dense(self.width)
        self.policy=nn.Dense(self.actions,kernel_init=nn.initializers.normal(.01))
        self.value=nn.Dense(len(SUPPORT),kernel_init=nn.initializers.zeros)
        self.reward=nn.Dense(len(SUPPORT),kernel_init=nn.initializers.zeros)
        self.continuation=nn.Dense(1,bias_init=nn.initializers.constant(4.6))
        self.decoder_in=nn.Dense(self.width)
        self.decoder=nn.Dense(MAP_DIM+160+6)
        self.project=nn.Dense(128)
        self.project_predict=nn.Dense(128)

    def represent(self,h,obs,previous_action,previous_reward,reset):
        h=jnp.where(reset[...,None],0.,h)
        previous_action=jnp.where(reset,0,previous_action)
        previous_reward=jnp.where(reset,0.,previous_reward)
        x=jnp.concatenate([obs,jax.nn.one_hot(previous_action,self.actions),symlog(previous_reward)[...,None]],-1)
        x=nn.relu(self.obs_layer(x))
        h,_=self.gru(h,x)
        return h

    def dynamics(self,h,action):
        x=jnp.concatenate([h,jax.nn.one_hot(action,self.actions)],-1)
        x=nn.relu(self.dyn_in(x))
        next_h=jnp.tanh(self.dyn_norm(h+self.dyn_out(x)))
        return next_h,self.reward(x),self.continuation(x)[...,0]

    def prediction(self,h):
        x=nn.relu(self.pred_layer(h))
        return self.policy(x),self.value(x)

    def observation(self,h):return self.decoder(nn.relu(self.decoder_in(h)))

    def projection(self,h,predict=False):
        x=self.project(h)
        return self.project_predict(nn.relu(x)) if predict else x

    def __call__(self,h,obs,action,reward,reset):
        z=self.represent(h,obs,action,reward,reset)
        predicted,r,c=self.dynamics(z,action)
        return self.prediction(z),self.observation(predicted),self.projection(predicted,True),self.projection(z),r,c


def observation_loss(logits,target):
    shape=target.shape[:-1]
    actual=target[...,:MAP_DIM].reshape(*shape,CELLS,BLOCKS+4)
    predicted=logits[...,:MAP_DIM].reshape(*shape,CELLS,BLOCKS+4)
    terrain=cross_entropy(predicted[...,:BLOCKS],actual[...,:BLOCKS]).mean(-1)
    mobs=optax.sigmoid_binary_cross_entropy(predicted[...,BLOCKS:],actual[...,BLOCKS:]).mean((-1,-2))
    counts=jnp.rint(target[...,MAP_DIM:MAP_DIM+16]*10).astype(jnp.int32).clip(0,9)
    count_logits=logits[...,MAP_DIM:MAP_DIM+160].reshape(*shape,16,10)
    inventory=cross_entropy(count_logits,jax.nn.one_hot(counts,10)).mean(-1)
    direction=cross_entropy(logits[...,MAP_DIM+160:MAP_DIM+164],target[...,MAP_DIM+16:MAP_DIM+20])
    light=jnp.square(jax.nn.sigmoid(logits[...,MAP_DIM+164])-target[...,MAP_DIM+20])
    sleeping=optax.sigmoid_binary_cross_entropy(logits[...,MAP_DIM+165],target[...,MAP_DIM+21])
    return terrain+mobs+inventory+direction+light+sleeping


def search(model,params,h,key,simulations=32,gumbel_scale=1.,absolute_q=False):
    logits,value=model.apply(params,h,method=model.prediction)
    def recurrent(p,key,actions,embedding):
        z,r,c=model.apply(p,embedding,actions,method=model.dynamics)
        pi,v=model.apply(p,z,method=model.prediction)
        return mctx.RecurrentFnOutput(reward=scalar_decode(r),discount=.99*jax.nn.sigmoid(c),
            prior_logits=pi,value=scalar_decode(v)),z
    result=mctx.gumbel_muzero_policy(params,key,mctx.RootFnOutput(prior_logits=logits,value=scalar_decode(value),embedding=h),
        recurrent,num_simulations=simulations,max_num_considered_actions=min(8,simulations),
        max_depth=5,gumbel_scale=gumbel_scale,
        qtransform=functools.partial(mctx.qtransform_completed_by_mix_value,rescale_values=not absolute_q))
    return result.action,result.action_weights,result.search_tree.summary().value


def encode_sequence(model,params,initial,batch):
    def body(h,x):
        obs,pa,pr,reset=x
        h=model.apply(params,h,obs,pa,pr,reset,method=model.represent)
        return h,h
    return jax.lax.scan(body,initial,(batch['obs'],batch['prev_action'],batch['prev_reward'],batch['reset']))[1]


def bootstrap_returns(rewards,dones,bootstrap,gamma=.99):
    """Per-start n-step targets; include terminal reward, zero later bootstrap."""
    result=jnp.zeros_like(bootstrap);discount=jnp.ones_like(bootstrap)
    for k in range(rewards.shape[0]):
        result=result+discount*rewards[k]
        discount=discount*gamma*(1-dones[k])
    return result+discount*bootstrap


def training_loss(model,params,target_params,batch,burnin=64,unroll=5,td=5,real_sequence_training=False):
    target_latents=jax.lax.stop_gradient(encode_sequence(model,target_params,batch['initial_h'],batch))
    # Refresh bootstrap values under the EMA target instead of using stale replay values.
    _,target_values=model.apply(target_params,target_latents,method=model.prediction)
    target_values=scalar_decode(target_values)
    prefix={k:v[:burnin] for k,v in batch.items() if k in ('obs','prev_action','prev_reward','reset')}
    h=jax.lax.stop_gradient(encode_sequence(model,params,batch['initial_h'],prefix)[-1])
    if real_sequence_training:
        sequence={k:v[burnin:burnin+unroll+1] for k,v in batch.items()
                  if k in ('obs','prev_action','prev_reward','reset')}
        posterior=encode_sequence(model,params,h,sequence)
        h=posterior[0]
    else:
        h=model.apply(params,h,batch['obs'][burnin],batch['prev_action'][burnin],
            batch['prev_reward'][burnin],batch['reset'][burnin],method=model.represent)
    root_h=h
    active=jnp.ones(h.shape[0]);losses={k:jnp.zeros_like(active) for k in ('policy','value','reward','termination','observation','consistency')}
    errors=[]
    for k in range(unroll+1):
        t=burnin+k
        pi,v=model.apply(params,h,method=model.prediction)
        target=bootstrap_returns(batch['reward'][t:t+td],batch['done'][t:t+td],target_values[t+td])
        if k==0:
            target=jnp.where(batch['root_reanalysed'],.5*target+.5*batch['root_search_value'],target)
        policy_loss=cross_entropy(pi,batch['policy'][t])
        value_loss=cross_entropy(v,scalar_target(target))
        if real_sequence_training:
            actual_pi,actual_v=model.apply(params,posterior[k],method=model.prediction)
            policy_loss=.5*(policy_loss+cross_entropy(actual_pi,batch['policy'][t]))
            value_loss=.5*(value_loss+cross_entropy(actual_v,scalar_target(target)))
        losses['policy']+=active*policy_loss
        losses['value']+=active*value_loss
        errors.append(jnp.abs(scalar_decode(v)-target)*active)
        if k==unroll:break
        next_h,r,c=model.apply(params,h,batch['action'][t],method=model.dynamics)
        losses['reward']+=active*cross_entropy(r,scalar_target(batch['reward'][t]))
        losses['termination']+=active*optax.sigmoid_binary_cross_entropy(c,1-batch['done'][t])
        losses['observation']+=active*observation_loss(model.apply(params,next_h,method=model.observation),batch['next_obs'][t])
        p=model.apply(params,next_h,True,method=model.projection)
        target_p=model.apply(target_params,target_latents[t+1],False,method=model.projection)
        p=p/jnp.maximum(jnp.linalg.norm(p,axis=-1,keepdims=True),1e-5)
        target_p=jax.lax.stop_gradient(target_p/jnp.maximum(jnp.linalg.norm(target_p,axis=-1,keepdims=True),1e-5))
        losses['consistency']+=active*(1-batch['done'][t])*(1-(p*target_p).sum(-1))
        active*=1-batch['done'][t]
        h=scale_gradient(next_h,.5)
    losses={k:v/(unroll+1 if k in ('policy','value') else unroll) for k,v in losses.items()}
    total=losses['policy']+.5*losses['value']+losses['reward']+losses['termination']+losses['observation']+2*losses['consistency']
    # TD replay priorities; clipping avoids numerical runaway in the sampling tree.
    priorities=jnp.clip(jnp.max(jnp.stack(errors),axis=0),1e-3,100.)
    return jnp.mean(batch['weights']*total),({k:v.mean() for k,v in losses.items()},priorities,root_h)

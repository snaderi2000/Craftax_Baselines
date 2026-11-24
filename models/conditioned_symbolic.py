"""Spatial symbolic PPO with internal-state-conditioned residual features.

Uses the supplied observation layout, not the meanings of individual tile or
achievement IDs. The 22 flat features condition generic learned spatial filters.
"""
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import distrax
from models.actor_critic import ActorCritic


def symbolic_spatial_features(obs, channels=64, blocks=3):
    """Called inside a compact module; preserves arbitrary leading dimensions."""
    shape=obs.shape[:-1];x=obs.reshape((-1,1345))
    grid=x[:,:1323].reshape((-1,7,9,21));stats=x[:,1323:]
    yy,xx=jnp.meshgrid(jnp.linspace(-1,1,7),jnp.linspace(-1,1,9),indexing='ij')
    position=jnp.broadcast_to(jnp.stack([yy,xx],-1),(len(x),7,9,2))
    grid=jnp.concatenate([grid,position],-1)
    context=nn.relu(nn.Dense(128,name='internal_context')(stats))
    h=nn.relu(nn.Conv(channels,(3,3),padding='SAME',name='map_input')(grid))
    for i in range(blocks):
        modulation=nn.Dense(2*channels,kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,name=f'condition_{i}')(context)
        scale,shift=jnp.split(modulation,2,-1)
        z=nn.LayerNorm(name=f'map_norm_{i}')(h)
        z=z*(1+jnp.tanh(scale[:,None,None,:]))+shift[:,None,None,:]
        z=nn.Conv(channels,(3,3),padding='SAME',name=f'map_block_{i}')(nn.relu(z))
        h=nn.relu(h+z/np.sqrt(blocks))
    return jnp.concatenate([h.reshape((len(x),-1)),stats,context],-1),shape


def symbolic_map_features(obs, view_size=33):
    """Encode current observation plus observed-terrain memory, no pixel input."""
    expected = 1345+2*view_size**2
    if obs.shape[-1] != expected:
        raise ValueError(f'Map observation has {obs.shape[-1]} entries; expected {expected}')
    features, shape = symbolic_spatial_features(obs[..., :1345])
    flat = obs.reshape((-1, expected))
    count = view_size**2
    terrain = flat[:, 1345:1345+count].reshape((-1, view_size, view_size)).astype(jnp.int32)
    age = flat[:, 1345+count:].reshape((-1, view_size, view_size, 1))
    yy, xx = jnp.meshgrid(jnp.linspace(-1, 1, view_size),
                          jnp.linspace(-1, 1, view_size), indexing='ij')
    coords = jnp.broadcast_to(jnp.stack((yy, xx), -1), (len(flat), view_size, view_size, 2))
    remembered = jnp.concatenate((jax.nn.one_hot(terrain, 18), age, coords), axis=-1)
    for i, channels in enumerate((16, 32, 32)):
        remembered = nn.relu(nn.Conv(channels, (3, 3), strides=(2, 2),
            padding='SAME', name=f'memory_conv_{i}')(remembered))
    return jnp.concatenate((features, remembered.reshape((len(flat), -1))), -1), shape


class ConditionedSymbolicActorCritic(nn.Module):
    action_dim:int=17
    channels:int=64
    width:int=256
    blocks:int=3

    @nn.compact
    def __call__(self,obs):
        features,shape=symbolic_spatial_features(obs,self.channels,self.blocks)
        actor=nn.relu(nn.Dense(self.width,name='actor_0')(features))
        actor=nn.relu(nn.Dense(self.width,name='actor_1')(actor))
        logits=nn.Dense(self.action_dim,kernel_init=nn.initializers.normal(.01),name='actor_out')(actor)
        critic=nn.relu(nn.Dense(self.width,name='critic_0')(features))
        critic=nn.relu(nn.Dense(self.width,name='critic_1')(critic))
        value=nn.Dense(1,name='critic_out')(critic)[...,0]
        return distrax.Categorical(logits=logits.reshape((*shape,self.action_dim))),value.reshape(shape)


def make_network(architecture):
    if architecture=='mlp':return ActorCritic(17,512,'tanh')
    if architecture=='conditioned':return ConditionedSymbolicActorCritic()
    raise ValueError(architecture)

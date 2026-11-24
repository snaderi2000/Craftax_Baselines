"""Opt-in 64x64 spatial observations for transfer from the local motion GRU."""
from functools import partial
from pathlib import Path
import flax.linen as nn
from flax.core import freeze, unfreeze, FrozenDict
import jax
import jax.numpy as jnp
from gymnax.environments import spaces
from symbolic_map_memory import simulator_state
from wrappers import GymnaxWrapper

SIZE = 64
LOCAL = 1364  # Original 1345 symbolic inputs and 19 motion inputs.


def augment(obs, state, mode):
    raw = simulator_state(state)
    if raw.map.shape != (SIZE, SIZE) or obs.shape != (LOCAL,):
        raise ValueError('World view requires Classic and the local motion observation')
    # Zero is unknown. Low five bits encode terrain ID+1; high four encode
    # independent occupancy flags (including overlapping creatures/projectiles).
    tiles = raw.map.astype(jnp.int32) + 1
    pos = raw.player_position
    local = ((abs(jnp.arange(SIZE)[:, None]-pos[0])<=3) &
             (abs(jnp.arange(SIZE)[None, :]-pos[1])<=4))
    if mode == 'local':
        tiles = jnp.where(local, tiles, 0)
    elif mode == 'entities':
        for bit, group in enumerate((raw.zombies,raw.cows,raw.skeletons,raw.arrows)):
            row,col = group.position.T
            occupied = jnp.zeros_like(tiles).at[row,col].max(group.mask.astype(jnp.int32))
            tiles = tiles + occupied * (32 << bit)
    elif mode != 'terrain':
        raise ValueError(f'Invalid view: {mode}')
    return jnp.concatenate((obs,tiles.reshape(-1),pos)).astype(jnp.float32)


class WorldViewWrapper(GymnaxWrapper):
    def __init__(self, env, mode):
        super().__init__(env)
        self.mode=mode

    def observation_space(self,params=None):
        return spaces.Box(-jnp.inf,jnp.inf,(LOCAL+SIZE**2+2,),jnp.float32)

    @partial(jax.jit,static_argnums=(0,2))
    def reset(self,key,params=None):
        obs,state=self._env.reset(key,params)
        return augment(obs,state,self.mode),state

    @partial(jax.jit,static_argnums=(0,4))
    def step(self,key,state,action,params=None):
        obs,state,r,d,info=self._env.step(key,state,action,params)
        return augment(obs,state,self.mode),state,r,d,info


class WorldReader(nn.Module):
    width: int

    @nn.compact
    def __call__(self, suffix):
        lead = suffix.shape[:-1]
        packed = suffix[...,:-2].astype(jnp.int32).reshape((*lead,SIZE,SIZE))
        pos = suffix[...,-2:]/(SIZE-1)
        terrain = nn.Embed(18,4,name='terrain')(packed & 31)
        entities = ((packed[...,None] >> jnp.arange(5,9)) & 1).astype(jnp.float32)
        features = jnp.concatenate((terrain,entities),-1)
        # Shared learned patch projection retains within-patch layout and the
        # flattened output retains the identity of each patch on the board.
        patches = features.reshape((*lead,8,8,8,8,8)).swapaxes(-4,-3)
        patches = nn.relu(nn.Dense(16,name='patch')(patches.reshape((*lead,64,512))))
        features = jnp.concatenate((patches.reshape((*lead,1024)),pos),-1)
        features = nn.relu(nn.Dense(64,name='board')(features))
        return nn.Dense(self.width,kernel_init=nn.initializers.zeros,
                        bias_init=nn.initializers.zeros,name='residual')(features)


def transplant(target, source):
    """Copy every old parameter, leaving only the new reader at initialization."""
    new=unfreeze(target)
    old=unfreeze(source)
    if set(new['params']) != set(old['params']) | {'world_reader'} or 'world_reader' in old['params']:
        raise ValueError('Expected a local GRU checkpoint and an expanded motion GRU')
    for name,value in old['params'].items():
        expected=new['params'][name]
        if jax.tree_util.tree_structure(value)!=jax.tree_util.tree_structure(expected):
            raise ValueError(f'Checkpoint structure mismatch in {name}')
        for x,y in zip(jax.tree_util.tree_leaves(expected),jax.tree_util.tree_leaves(value)):
            if x.shape!=y.shape or x.dtype!=y.dtype:raise ValueError(f'Checkpoint mismatch in {name}')
        new['params'][name]=value
    return freeze(new) if isinstance(target,FrozenDict) else new


def source_params(path):
    from orbax.checkpoint import PyTreeCheckpointer
    p=Path(path).resolve()
    return PyTreeCheckpointer().restore(str(p/'default' if (p/'default').is_dir() else p))['params']

"""Resource addresses and episode events derived from observed terrain and factual motion."""
from functools import partial

from flax import struct
import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import Action, BlockType, DIRECTIONS
from gymnax.environments import spaces
from wrappers import GymnaxWrapper

MODES = ('masked', 'locations', 'events')
SLOTS = 6
FIELDS = 5
RESOURCE_CODES = tuple(x.value+1 for x in (BlockType.DIAMOND, BlockType.WATER, BlockType.PLANT, BlockType.RIPE_PLANT))


@struct.dataclass
class LandmarkState:
    env_state: object
    entered: object
    planted_at: object
    planted_step: object


def landmark_features(s, mode):
    if mode == 'masked':
        return jnp.zeros(SLOTS*FIELDS, jnp.float32)
    m = s.env_state
    rr, cc = jnp.indices(m.terrain.shape)
    dr, dc = rr-m.position[0], cc-m.position[1]
    dist = abs(dr)+abs(dc)
    def record(mask, cost, times):
        idx = jnp.argmin(jnp.where(mask, cost, 100000))
        age = jnp.log1p(jnp.maximum(m.elapsed-times.ravel()[idx], 0))/jnp.log(10001.)
        row = jnp.array([1., dr.ravel()[idx]/63., dc.ravel()[idx]/63., dist.ravel()[idx]/126., age])
        return jnp.where(mask.any(), row, 0.)
    slots = [record(m.terrain==code, dist, m.last_seen) for code in RESOURCE_CODES]
    extra = jnp.zeros((2,FIELDS), jnp.float32)
    if mode == 'events':
        pr, pc = s.planted_at
        planted = (rr==pr)&(cc==pc)&(s.planted_step>=0)&((m.terrain==16)|(m.terrain==17))
        last_plant = record(planted, dist, jnp.broadcast_to(s.planted_step,dist.shape))
        # Prefer a known frontier in the least physically explored 8x8 sector.
        # Outside the board is never a frontier.
        unknown = jnp.pad(m.terrain==0, 1)
        frontier = (unknown[:-2,1:-1]|unknown[2:,1:-1]|unknown[1:-1,:-2]|unknown[1:-1,2:])
        walkable = (m.terrain==3)|(m.terrain==7)|(m.terrain==8)|(m.terrain==14)
        counts = s.entered.reshape(8,8,8,8).sum((1,3))
        frontier_slot = record(frontier&walkable, 128*counts[rr//8,cc//8]+dist, m.last_seen)
        extra = jnp.stack((last_plant,frontier_slot))
    return jnp.concatenate((jnp.stack(slots),extra)).reshape(-1)


class LandmarkWrapper(GymnaxWrapper):
    def __init__(self, env, mode):
        super().__init__(env)
        if mode not in MODES:
            raise ValueError('Invalid landmark mode')
        self.mode = mode

    def observation_space(self, params=None):
        size = self._env.observation_space(params).shape[0]+SLOTS*FIELDS
        return spaces.Box(-1., 17., (size,), jnp.float32)

    @partial(jax.jit, static_argnums=(0,2))
    def reset(self, key, params=None):
        obs, memory = self._env.reset(key, params)
        entered = jnp.zeros_like(memory.terrain, dtype=bool).at[tuple(memory.position)].set(True)
        s = LandmarkState(memory, entered, jnp.zeros(2,jnp.int32), jnp.int32(-1))
        return jnp.concatenate((obs,landmark_features(s,self.mode))), s

    @partial(jax.jit, static_argnums=(0,4))
    def step(self, key, s, action, params=None):
        obs, m, reward, done, info = self._env.step(key, s.env_state, action, params)
        old = s.env_state
        front = jnp.clip(old.position+DIRECTIONS[old.env_state.player_direction],0,63)
        at = tuple(front)
        planted = (action==Action.PLACE_PLANT.value)&(old.terrain[at]!=16)&(m.terrain[at]==16)
        s = LandmarkState(m, s.entered.at[tuple(m.position)].set(True),
                          jnp.where(planted,front,s.planted_at),jnp.where(planted,m.elapsed,s.planted_step))
        return jnp.concatenate((obs,landmark_features(s,self.mode))), s, reward, done, info

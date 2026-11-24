"""Optional survival context derived only from observations and existing odometry.

The global route uses remembered observed terrain, never unseen simulator tiles.
It is a static walking route to a water edge, not a safety guarantee: monsters,
mining, and unobserved changes are not planned. Paths longer than 128 moves are
reported as unavailable. Masked mode is a parameter-matched ablation.
"""
from functools import partial
from flax import struct
import jax
import jax.numpy as jnp
from gymnax.environments import spaces
from wrappers import GymnaxWrapper

FEATURE_SIZE=40
ROUTE_HORIZON=128
DIRECTIONS=((0,0),(0,-1),(0,1),(-1,0),(1,0))


def adjacent(x,fill):
    p=jnp.pad(x,1,constant_values=fill)
    return jnp.stack((p[1:-1,:-2],p[1:-1,2:],p[:-2,1:-1],p[2:,1:-1]))


def water_route_features(terrain,position,local):
    # Memory encoding is tile ID + 1; zero is unobserved.
    water=terrain==4
    walkable=(terrain==3)|(terrain==7)|(terrain==8)|(terrain==14)
    goals=walkable&adjacent(water,False).any(0)
    infinity=ROUTE_HORIZON+1
    initial=jnp.where(goals,0,infinity).astype(jnp.uint8)
    def relax(_,dist):
        near=adjacent(dist,infinity).min(0)
        return jnp.where(walkable,jnp.minimum(dist,jnp.minimum(near+1,infinity)),infinity).astype(jnp.uint8)
    distances=jax.lax.fori_loop(0,ROUTE_HORIZON,relax,initial)
    coords=position[None]+jnp.asarray(DIRECTIONS,jnp.int32)
    valid=(coords[:,0]>=0)&(coords[:,0]<terrain.shape[0])&(coords[:,1]>=0)&(coords[:,1]<terrain.shape[1])
    at=jnp.where(valid,distances[coords[:,0].clip(0,terrain.shape[0]-1),coords[:,1].clip(0,terrain.shape[1]-1)],infinity)
    reachable=at<=ROUTE_HORIZON
    water_adjacent=jnp.where(valid[1:],water[coords[1:,0].clip(0,terrain.shape[0]-1),coords[1:,1].clip(0,terrain.shape[1]-1)],False)
    facing=jnp.argmax(local[1339:1343])
    return jnp.concatenate((jnp.minimum(at.astype(jnp.float32)/ROUTE_HORIZON,1.),reachable.astype(jnp.float32),
        water_adjacent.astype(jnp.float32),jnp.array([water_adjacent[facing],water.any()],jnp.float32)))


@struct.dataclass
class SurvivalMapState:
    env_state:object
    depleted_steps:object
    last_sleeping:object
    blocked_streak:object


class ClassicSurvivalMapWrapper(GymnaxWrapper):
    def __init__(self,env,mode='enabled'):
        super().__init__(env)
        if mode not in ('enabled','masked'):raise ValueError('Survival map mode must be enabled or masked')
        self.mode=mode

    def observation_space(self,params=None):
        return spaces.Box(-1.,17.,(self._env.observation_space(params).shape[0]+FEATURE_SIZE,),jnp.float32)

    def augment(self,obs,state,action=None,displacement=None,blocked=False):
        if self.mode=='masked':features=jnp.zeros(FEATURE_SIZE)
        else:
            routes=water_route_features(state.env_state.terrain,state.env_state.position,obs[:1345])
            features=jnp.concatenate((routes,state.depleted_steps.astype(jnp.float32)/256.,
                jnp.zeros(17) if action is None else jax.nn.one_hot(action,17),
                jnp.zeros(2) if displacement is None else displacement.astype(jnp.float32),
                jnp.array([blocked,state.blocked_streak/64.],jnp.float32)))
        return jnp.concatenate((obs,features))

    def initialize(self,obs,map_state):
        state=SurvivalMapState(map_state,(obs[1336:1339]<=0).astype(jnp.int32),obs[1344]>0,jnp.int32(0))
        return self.augment(obs,state),state

    @partial(jax.jit,static_argnums=(0,2))
    def reset(self,key,params=None):
        obs,state=self._env.reset(key,params)
        return self.initialize(obs,state)

    @partial(jax.jit,static_argnums=(0,4))
    def step(self,key,state,action,params=None):
        obs,memory,reward,done,info=self._env.step(key,state.env_state,action,params)
        displacement=memory.position-state.env_state.position
        blocked=(action>=1)&(action<=4)&jnp.all(displacement==0)&~state.last_sleeping
        updated=SurvivalMapState(memory,jnp.where(obs[1336:1339]<=0,jnp.minimum(state.depleted_steps+1,256),0),
            obs[1344]>0,jnp.where(blocked,jnp.minimum(state.blocked_streak+1,64),0))
        return self.augment(obs,updated,action,displacement,blocked),updated,reward,done,info

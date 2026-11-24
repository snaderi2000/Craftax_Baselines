"""Observed stone-attempt labels, balanced replay, and an actor-shared predictor."""
import jax
import jax.numpy as jnp
from wrappers import GymnaxWrapper
from craftax.craftax_classic.constants import BlockType, Action, DIRECTIONS


class StoneOutcomeWrapper(GymnaxWrapper):
    def step(self, key, state, action, params=None):
        front = state.player_position + DIRECTIONS[state.player_direction]
        stone = state.map[front[0], front[1]] == BlockType.STONE.value
        attempt = stone & (action == Action.DO.value)
        actionable = stone & (state.inventory.wood_pickaxe > 0) & ~state.is_sleeping & ~state.mob_map[front[0],front[1]]
        obs, next_state, reward, done, info = self._env.step(key,state,action,params)
        success = next_state.inventory.stone > state.inventory.stone
        info.update(stone_attempt=attempt, stone_success=attempt & success,
                    stone_actionable=actionable, stone_has_pickaxe=state.inventory.wood_pickaxe > 0,
                    stone_sleeping=state.is_sleeping,stone_blocked=state.mob_map[front[0],front[1]],
                    stone_inventory=state.inventory.stone)
        return obs,next_state,reward,done,info


def initialize_head(params, key, width):
    params = dict(params)
    params['params'] = dict(params['params'])
    params['params']['StoneOutcome'] = dict(kernel=jax.random.normal(key,(width+17,))*0.01,
                                           bias=jnp.asarray(0.))
    return params


def predict_logits(params, observations, actions):
    root = params['params']
    h = observations
    for i in range(3):
        h = jnp.tanh(h @ root[f'Dense_{i}']['kernel']+root[f'Dense_{i}']['bias'])
    features = jnp.concatenate([h,jax.nn.one_hot(actions,17)],axis=-1)
    return features @ root['StoneOutcome']['kernel'] + root['StoneOutcome']['bias']


def empty_replay(capacity, obs_dim):
    return dict(obs=jnp.zeros((2,capacity,obs_dim)), actions=jnp.zeros((2,capacity),jnp.int32),
                counts=jnp.zeros(2,jnp.int32), totals=jnp.zeros(2,jnp.int32))


def append_replay(buffer, observations, actions, eligible, success):
    capacity = buffer['obs'].shape[1]
    result = dict(buffer)
    k = min(capacity,len(actions))
    for label in range(2):
        mask = eligible & (success == bool(label))
        ranks = jnp.where(mask,jnp.arange(len(actions)),-1)
        values, indices = jax.lax.top_k(ranks,k)
        valid = values >= 0
        # At most k latest attempts of each class enter the ring per rollout.
        slots = (buffer['totals'][label]+jnp.arange(k)) % capacity
        slots = jnp.where(valid,slots,capacity)
        row = buffer['obs'][label].at[slots].set(observations[indices],mode='drop')
        acts = buffer['actions'][label].at[slots].set(actions[indices],mode='drop')
        result['obs'] = result['obs'].at[label].set(row)
        result['actions'] = result['actions'].at[label].set(acts)
        inserted = valid.sum()
        result['totals'] = result['totals'].at[label].add(inserted)
        result['counts'] = result['counts'].at[label].set(jnp.minimum(capacity,buffer['counts'][label]+inserted))
    return result


def sample_replay(buffer, key, batch_size):
    available = buffer['counts'] > 0
    labels = jnp.arange(batch_size) % 2
    labels = jnp.where(available.all(),labels,jnp.where(available[1],1,0))
    indices = jax.random.randint(key,(batch_size,),0,jnp.maximum(buffer['counts'][labels],1))
    return (buffer['obs'][labels,indices],buffer['actions'][labels,indices],labels.astype(jnp.float32),
            available.any()*jnp.ones(batch_size))


def outcome_loss(params, batch):
    obs,actions,labels,mask = batch
    logits = predict_logits(params,obs,actions)
    ce = jnp.maximum(logits,0)-logits*labels+jnp.log1p(jnp.exp(-jnp.abs(logits)))
    return (ce*mask).sum()/jnp.maximum(mask.sum(),1)

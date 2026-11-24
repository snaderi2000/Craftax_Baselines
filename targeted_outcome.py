"""Stone-style relevant-attempt prediction for all 22 Classic tasks."""
import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import Achievement, Action, BlockType, DIRECTIONS
from wrappers import GymnaxWrapper
from multi_outcome import append_replay
from multi_outcome import empty_replay as _empty_replay

TASKS = tuple(a.name.lower() for a in Achievement)
N_TASKS = len(TASKS)
RESOURCE_TASKS = {
    'collect_wood': ('wood', BlockType.TREE), 'collect_stone': ('stone', BlockType.STONE),
    'collect_coal': ('coal', BlockType.COAL), 'collect_iron': ('iron', BlockType.IRON),
    'collect_diamond': ('diamond', BlockType.DIAMOND), 'collect_sapling': ('sapling', BlockType.GRASS)}
CRAFT_TASKS = {f'make_{material}_{tool}': (f'{material}_{tool}',getattr(Action,f'MAKE_{material.upper()}_{tool.upper()}'))
               for material in ('wood','stone','iron') for tool in ('pickaxe','sword')}
PLACE_TASKS = {'place_table': (Action.PLACE_TABLE,BlockType.CRAFTING_TABLE),
               'place_furnace': (Action.PLACE_FURNACE,BlockType.FURNACE),
               'place_stone': (Action.PLACE_STONE,BlockType.STONE),
               'place_plant': (Action.PLACE_PLANT,BlockType.PLANT)}


def attempt_labels(before, after, action):
    front = before.player_position + DIRECTIONS[before.player_direction]
    tile = before.map[front[0],front[1]]
    next_tile = after.map[front[0],front[1]]
    earned = after.achievements & ~before.achievements
    doing = action == Action.DO.value
    labels, eligible = [], []
    for i,name in enumerate(TASKS):
        success = earned[i]
        if name in RESOURCE_TASKS:
            field, block = RESOURCE_TASKS[name]
            relevant = doing & (tile == block.value)
            success = getattr(after.inventory,field) > getattr(before.inventory,field)
        elif name in CRAFT_TASKS:
            field, craft_action = CRAFT_TASKS[name]
            relevant = action == craft_action.value
            success = getattr(after.inventory,field) > getattr(before.inventory,field)
        elif name in PLACE_TASKS:
            place_action, block = PLACE_TASKS[name]
            relevant = action == place_action.value
            success = (tile != next_tile) & (next_tile == block.value)
            success = success | earned[i]
        else:
            # Unambiguous first-event labels; exclude already-earned flags.
            if name in ('eat_cow','defeat_zombie','defeat_skeleton'):
                mob_name = {'eat_cow':'cows','defeat_zombie':'zombies','defeat_skeleton':'skeletons'}[name]
                mobs = getattr(before,mob_name)
                relevant = doing & jnp.any(mobs.mask & jnp.all(mobs.position == front,axis=-1))
            elif name == 'collect_drink':
                relevant = doing & (tile == BlockType.WATER.value)
            elif name == 'eat_plant':
                relevant = doing & ((tile == BlockType.PLANT.value) | (tile == BlockType.RIPE_PLANT.value))
            elif name == 'wake_up':
                relevant = before.is_sleeping | (action == Action.SLEEP.value)
            else:
                raise ValueError(f'Unmapped task {name}')
            relevant = relevant & ~before.achievements[i]
        eligible.append(relevant)
        labels.append(success)
    return jnp.stack(eligible),jnp.stack(labels)


class TargetedOutcomeWrapper(GymnaxWrapper):
    def step(self,key,state,action,params=None):
        obs,next_state,reward,done,info = self._env.step(key,state,action,params)
        eligible,success = attempt_labels(state,next_state,action)
        for i,name in enumerate(TASKS):
            info['targeted/eligible/'+name] = eligible[i]
            info['targeted/success/'+name] = success[i] & eligible[i]
        return obs,next_state,reward,done,info


def initialize_head(params,key,width):
    params = dict(params)
    params['params'] = dict(params['params'])
    params['params']['TargetedOutcome'] = dict(kernel=jax.random.normal(key,(width+17,N_TASKS))*.01,
                                             bias=jnp.zeros(N_TASKS))
    return params


def predict_logits(params,obs,actions):
    root = params['params']
    h = obs
    for i in range(3):
        h = jnp.tanh(h @ root[f'Dense_{i}']['kernel'] + root[f'Dense_{i}']['bias'])
    features = jnp.concatenate([h,jax.nn.one_hot(actions,17)],axis=-1)
    return features @ root['TargetedOutcome']['kernel'] + root['TargetedOutcome']['bias']


def empty_replay(capacity,obs_dim):
    return _empty_replay(capacity,obs_dim,N_TASKS)


def sample_replay(buffer,key,per_task_batch):
    # Tile task IDs so every PPO minibatch receives an equal slice of every task.
    outputs = jnp.tile(jnp.arange(N_TASKS),per_task_batch)
    labels = jnp.repeat(jnp.arange(per_task_batch)%2,N_TASKS)
    counts = buffer['counts'].reshape(N_TASKS,2)
    labels = jnp.where(counts[outputs,labels]>0,labels,1-labels)
    queues = 2*outputs+labels
    indices = jax.random.randint(key,queues.shape,0,jnp.maximum(buffer['counts'][queues],1))
    return (buffer['obs'][queues,indices],buffer['actions'][queues,indices],outputs,
            labels.astype(jnp.float32),(buffer['counts'][queues]>0).astype(jnp.float32))


def outcome_loss(params,batch):
    obs,actions,outputs,labels,mask = batch
    logits = jnp.take_along_axis(predict_logits(params,obs,actions),outputs[:,None],axis=-1)[:,0]
    ce = jnp.maximum(logits,0)-logits*labels+jnp.log1p(jnp.exp(-jnp.abs(logits)))
    return (ce*mask).sum()/jnp.maximum(mask.sum(),1)

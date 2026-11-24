"""Achievement-ID practice from a policy's own natural pre-event simulator states."""
from functools import partial
from typing import Any
import gzip
import json
import pickle
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from craftax.craftax_classic.constants import Achievement
from craftax.craftax_classic.envs.common import compute_score
from wrappers import GymnaxWrapper

TASKS = tuple(a.name.lower() for a in Achievement)
OFFSETS = (1,5,20)


@struct.dataclass
class PracticeState:
    raw: Any
    history: Any
    archive: Any
    archive_counts: Any
    archive_totals: Any
    archive_lags: Any
    archive_workers: Any
    archive_episodes: Any
    archive_event_steps: Any
    cursor: Any
    history_count: Any
    target: Any
    elapsed: Any
    episode_return: Any
    episode_length: Any
    returned_return: Any
    returned_length: Any
    episode_serial: Any
    total_steps: Any
    practice_steps: Any
    attempts_by_id: Any
    successes_by_id: Any
    timeouts_by_id: Any
    rarity_window: Any = None
    rarity_sum: Any = None
    rarity_total: Any = None
    rarity_last_update: Any = None
    rarity_selected: Any = None
    rarity_selection_counts: Any = None
    reuse_counts: Any = None
    rarity_starts: Any = None
    rarity_unique_starts: Any = None
    backward_bank_counts: Any = None
    backward_bank_totals: Any = None
    backward_level: Any = None
    backward_outcomes: Any = None
    backward_attempt_counts: Any = None
    backward_last_check: Any = None
    backward_promotions: Any = None
    practice_level: Any = None
    practice_steps_by_id: Any = None


def choose(mask, yes, no):
    return jax.tree_util.tree_map(lambda a,b:jnp.where(
        mask.reshape(mask.shape+(1,)*(a.ndim-mask.ndim)),a,b),yes,no)


class IDPractice(GymnaxWrapper):
    """Same batched normal/reset machinery for baseline (0) and practice slots."""
    def __init__(self, base, num_envs=128, practice_envs=32, capacity=64, horizon=64, full_episode=False):
        super().__init__(base)
        self.num_envs,self.practice_envs,self.capacity,self.horizon = num_envs,practice_envs,capacity,horizon
        self.rare_four = False
        self.full_episode = full_episode
        self.backward = False
        self.offsets = OFFSETS
        self.history_length = max(self.offsets)
        self.assigned = jnp.arange(num_envs) >= num_envs-practice_envs
        self.step_fn = jax.vmap(base.step,in_axes=(0,0,0,None))
        self.reset_fn = jax.vmap(base.reset,in_axes=(0,None))
        self.obs_fn = jax.vmap(base.get_obs)

    @partial(jax.jit,static_argnums=(0,2))
    def reset(self,key,params=None):
        obs,raw = self.reset_fn(jax.random.split(key,self.num_envs),params)
        history = jax.tree_util.tree_map(lambda x:jnp.zeros((self.history_length,)+x.shape,x.dtype),raw)
        archive = jax.tree_util.tree_map(lambda x:jnp.zeros((22,self.capacity)+x.shape[1:],x.dtype),raw)
        zeros = jnp.zeros(self.num_envs,jnp.int32)
        state = PracticeState(raw,history,archive,jnp.zeros(22,jnp.int32),jnp.zeros(22,jnp.int32),
            jnp.zeros((22,self.capacity),jnp.int32),jnp.zeros((22,self.capacity),jnp.int32),
            jnp.zeros((22,self.capacity),jnp.int32),jnp.zeros((22,self.capacity),jnp.int32),
            jnp.asarray(0),zeros,jnp.full(self.num_envs,-1,jnp.int32),zeros,
            jnp.zeros(self.num_envs),zeros,jnp.zeros(self.num_envs),zeros,zeros,
            jnp.asarray(0),jnp.asarray(0),jnp.zeros(22,jnp.int32),jnp.zeros(22,jnp.int32),jnp.zeros(22,jnp.int32))
        return obs,state.replace(practice_steps_by_id=jnp.zeros(22,jnp.int32))

    def _record(self,state,next_raw):
        history = jax.tree_util.tree_map(lambda h,x:h.at[state.cursor].set(x),state.history,state.raw)
        offsets = len(self.offsets)
        indices = (state.cursor-jnp.asarray(self.offsets)+1)%self.history_length
        candidates = jax.tree_util.tree_map(lambda h:jnp.take(h,indices,axis=0).reshape(
            (offsets*self.num_envs,)+h.shape[2:]),history)
        new_events = next_raw.achievements & ~state.raw.achievements
        normal = state.target < 0
        valid = (state.history_count[None,:]+1 >= jnp.asarray(self.offsets)[:,None]) & normal[None,:]
        workers = jnp.tile(jnp.arange(self.num_envs),offsets)
        lags = jnp.repeat(jnp.asarray(self.offsets),self.num_envs)
        episodes = jnp.tile(state.episode_serial,offsets)
        steps = jnp.tile(next_raw.timestep,offsets)
        k = min(self.capacity,offsets*self.num_envs)
        initial = (state.archive,state.archive_counts,state.archive_totals,
                   state.archive_lags,state.archive_workers,state.archive_episodes,state.archive_event_steps)
        def task(i,carry):
            match = (valid & new_events[:,i][None,:]).reshape(-1) & ~candidates.achievements[:,i]
            def insert(carry):
                archive,counts,totals,lag,worker,episode,event_step = carry
                ranks,ix = jax.lax.top_k(jnp.where(match,jnp.arange(offsets*self.num_envs),-1),k)
                good = ranks >= 0
                slots = jnp.where(good,(totals[i]+jnp.arange(k))%self.capacity,self.capacity)
                archive = jax.tree_util.tree_map(lambda a,x:a.at[i].set(
                    a[i].at[slots].set(x[ix],mode='drop')),archive,candidates)
                def put(table,values):
                    return table.at[i].set(table[i].at[slots].set(values[ix],mode='drop'))
                n = good.sum()
                return (archive,counts.at[i].set(jnp.minimum(self.capacity,counts[i]+n)),totals.at[i].add(n),
                        put(lag,lags),put(worker,workers),put(episode,episodes),put(event_step,steps))
            return jax.lax.cond(match.any(),insert,lambda c:c,carry)
        values = jax.lax.fori_loop(0,22,task,initial)
        return state.replace(history=history,archive=values[0],archive_counts=values[1],
            archive_totals=values[2],archive_lags=values[3],archive_workers=values[4],
            archive_episodes=values[5],archive_event_steps=values[6])

    def _starts(self,key,normal,state):
        task_key,entry_key = jax.random.split(key)
        tasks = jax.random.categorical(task_key,jnp.where(state.archive_counts>0,0.,-1e9),
                                      shape=(self.num_envs,)).astype(jnp.int32)
        entries = jax.random.randint(entry_key,(self.num_envs,),0,jnp.maximum(state.archive_counts[tasks],1))
        saved = jax.tree_util.tree_map(lambda x:x[tasks,entries],state.archive)
        use = self.assigned & (state.archive_counts.sum()>0)
        return choose(use,saved,normal),jnp.where(use,tasks,-1)

    def _reset_choices(self,key,normal,state,done):
        starts,targets = self._starts(key,normal,state)
        return starts,targets,state

    def _time_limit(self,state):
        return self.horizon

    @partial(jax.jit,static_argnums=(0,4))
    def step(self,key,state,actions,params=None):
        step_key,reset_key,practice_key = jax.random.split(key,3)
        obs,next_raw,reward,terminal,info = self.step_fn(
            jax.random.split(step_key,self.num_envs),state.raw,actions,params)
        practice = state.target >= 0
        reached = practice & next_raw.achievements[jnp.arange(self.num_envs),jnp.maximum(state.target,0)]
        earned_now = reached & ~state.raw.achievements[jnp.arange(self.num_envs),jnp.maximum(state.target,0)]
        elapsed = state.elapsed+1
        if self.full_episode:
            truncated = jnp.zeros_like(terminal)
            done = terminal
            success = reached & done
        else:
            success = reached
            truncated = practice & (elapsed>=self._time_limit(state)) & ~terminal & ~success
            done = terminal | success | truncated
        if self.practice_envs:
            state = self._record(state,next_raw)
        if self.rare_four:
            state = self._update_rarity(state,next_raw,terminal & ~practice,practice_key)
        if self.backward:
            state = self._update_curriculum(state,done & practice,success)
        info.update(jax.vmap(compute_score)(next_raw,done))
        returns = state.episode_return+reward
        lengths = state.episode_length+1
        info.update(returned_episode=done & ~practice,
                    returned_episode_returns=jnp.where(done,returns,state.returned_return),
                    returned_episode_lengths=jnp.where(done,lengths,state.returned_length),
                    timestep=next_raw.timestep)
        info.update({'practice/is_practice':practice,'practice/success':success,
                     'practice/environment_terminal':terminal,
                     'practice/target_earned_now':earned_now,
                     'practice/truncated':truncated,'practice/ended':done & practice,
                     'practice/target':state.target,'practice/bootstrap_obs':obs})
        for i,name in enumerate(TASKS):
            target = state.target==i
            info['practice/attempt_end/'+name] = done & target
            info['practice/success_end/'+name] = success & target
            info['practice/archive_count/'+name] = jnp.full(self.num_envs,state.archive_counts[i])
        _,normal_reset = self.reset_fn(jax.random.split(reset_key,self.num_envs),params)
        if self.practice_envs:
            starts,targets,state = self._reset_choices(practice_key,normal_reset,state,done)
        else:
            starts,targets = normal_reset,jnp.full(self.num_envs,-1,jnp.int32)
        raw = choose(done,starts,next_raw)
        by_id = jax.nn.one_hot(jnp.maximum(state.target,0),22,dtype=jnp.int32)*practice[:,None]
        state = state.replace(raw=raw,cursor=(state.cursor+1)%self.history_length,
            history_count=jnp.where(done,0,jnp.minimum(self.history_length,state.history_count+1)),
            target=jnp.where(done,targets,state.target),elapsed=jnp.where(done,0,elapsed),
            episode_return=jnp.where(done,0.,returns),episode_length=jnp.where(done,0,lengths),
            returned_return=info['returned_episode_returns'],returned_length=info['returned_episode_lengths'],
            episode_serial=state.episode_serial+done.astype(jnp.int32),
            total_steps=state.total_steps+self.num_envs,practice_steps=state.practice_steps+practice.sum(),
            practice_steps_by_id=state.practice_steps_by_id+by_id.sum(axis=0),
            attempts_by_id=state.attempts_by_id+(done[:,None]*by_id).sum(axis=0),
            successes_by_id=state.successes_by_id+((success & done)[:,None]*by_id).sum(axis=0),
            timeouts_by_id=state.timeouts_by_id+(truncated[:,None]*by_id).sum(axis=0))
        # Render the exact chosen state after a reset; no inventory/map/flag edits.
        return self.obs_fn(raw),state,reward,done,info


def save_archive(state,directory):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    data = dict(states=state.archive,counts=state.archive_counts,
        totals=state.archive_totals,lags=state.archive_lags,workers=state.archive_workers,
        episodes=state.archive_episodes,event_steps=state.archive_event_steps)
    if state.backward_bank_counts is not None:
        per_level = state.archive_lags.shape[1]//4
        data['valid_mask'] = (jnp.arange(per_level)[None,None,:] < state.backward_bank_counts[:,:,None]).reshape(22,-1)
    payload = jax.device_get(data)
    with gzip.open(directory/'state_archive.pkl.gz','wb') as f:
        pickle.dump(payload,f,protocol=pickle.HIGHEST_PROTOCOL)
    offsets = (1,5,20,50) if state.backward_bank_counts is not None else OFFSETS
    summary = dict(tasks=TASKS,offsets=offsets,counts=np.asarray(payload['counts']).tolist(),
        inserted=np.asarray(payload['totals']).tolist(),
        total_steps=int(state.total_steps),practice_steps=int(state.practice_steps),
        practice_steps_by_id=np.asarray(state.practice_steps_by_id).tolist(),
        practice_fraction=float(state.practice_steps/jnp.maximum(state.total_steps,1)),
        attempts_by_id=np.asarray(state.attempts_by_id).tolist(),
        successes_by_id=np.asarray(state.successes_by_id).tolist(),timeouts_by_id=np.asarray(state.timeouts_by_id).tolist(),
        note='Full pre-event simulator states from this run only; grouped by achievement ID. Load only trusted local pickle files.')
    if state.reuse_counts is not None:
        extra = jax.device_get(dict(reuse_counts=state.reuse_counts,normal_window=state.rarity_window,
            normal_sum=state.rarity_sum,normal_episodes=state.rarity_total,selected=state.rarity_selected,
            selection_counts=state.rarity_selection_counts,starts=state.rarity_starts,
            unique_snapshot_starts=state.rarity_unique_starts))
        with gzip.open(directory/'rarity_and_reuse.pkl.gz','wb') as f:
            pickle.dump(extra,f,protocol=pickle.HIGHEST_PROTOCOL)
        diversity,used_diversity = [],[]
        for i,count in enumerate(payload['counts']):
            valid = payload['valid_mask'][i] if 'valid_mask' in payload else np.arange(payload['lags'].shape[1])<int(count)
            origins = np.stack([payload['workers'][i,valid],payload['episodes'][i,valid]],axis=-1)
            diversity.append(len(np.unique(origins,axis=0)))
            used_diversity.append(len(np.unique(origins[extra['reuse_counts'][i,valid]>0],axis=0)))
        summary.update(normal_episodes=int(extra['normal_episodes']),
            normal_window_size=int(state.rarity_window.shape[0]),
            final_normal_rates=(np.asarray(extra['normal_sum'])/max(1,min(int(extra['normal_episodes']),state.rarity_window.shape[0]))).tolist(),
            final_selected_ids=np.flatnonzero(extra['selected']).tolist(),
            selection_counts=np.asarray(extra['selection_counts']).tolist(),
            starts_by_id=np.asarray(extra['starts']).tolist(),
            unique_snapshot_starts_by_id=np.asarray(extra['unique_snapshot_starts']).tolist(),
            archived_source_episodes_by_id=diversity,used_current_source_episodes_by_id=used_diversity,
            max_current_snapshot_reuse=int(np.asarray(extra['reuse_counts']).max()))
    if state.backward_level is not None:
        skill = jax.device_get(dict(level=state.backward_level,bank_counts=state.backward_bank_counts,
            attempts=state.backward_attempt_counts,outcomes=state.backward_outcomes,promotions=state.backward_promotions))
        with gzip.open(directory/'curriculum.pkl.gz','wb') as f:
            pickle.dump(skill,f,protocol=pickle.HIGHEST_PROTOCOL)
        summary.update(curriculum_level=np.asarray(skill['level']).tolist(),
            curriculum_bank_counts=np.asarray(skill['bank_counts']).tolist(),
            curriculum_attempts=np.asarray(skill['attempts']).tolist(),
            curriculum_promotions=np.asarray(skill['promotions']).tolist())
    (directory/'archive_summary.json').write_text(json.dumps(summary,indent=2)+'\n')

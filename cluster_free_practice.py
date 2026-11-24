"""Label-free simulator-start practice; no achievement-ID selection or grouping.

One uniformly selected five-step-pre-reward start per completed NORMAL episode
enters a FIFO bank. Practice never ends just because an achievement is earned.
"""
from functools import partial
from typing import Any
from pathlib import Path
import gzip
import json
import pickle
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from wrappers import GymnaxWrapper
from id_practice import choose


def sampling_probabilities(score, usable, prioritized, key, uniform_fraction=.2):
    count = jnp.maximum(usable.sum(), 1)
    uniform = usable.astype(jnp.float32) / count
    if not prioritized:
        return uniform
    # Random tie breaking: equal initial scores do not favor bank position.
    jitter = jax.random.uniform(key, score.shape) * 1e-6
    order = jnp.argsort(jnp.where(usable, score + jitter, -jnp.inf))
    ranks = jnp.zeros_like(score).at[order].set(jnp.arange(len(score), dtype=jnp.float32))
    ranks = jnp.where(usable, ranks - (len(score)-usable.sum()) + 1, 0)
    rank_probability = ranks / jnp.maximum(ranks.sum(), 1)
    return uniform_fraction * uniform + (1-uniform_fraction) * rank_probability


@struct.dataclass
class ClusterFreeState:
    raw: Any
    history: Any
    candidate: Any
    candidate_count: Any
    history_count: Any
    cursor: Any
    bank: Any
    bank_info: Any
    bank_total: Any
    entry: Any
    generation: Any
    elapsed: Any
    episode_serial: Any
    episode_return: Any
    episode_length: Any
    score_return: Any
    score_discount: Any
    score_start_value: Any
    total_steps: Any
    practice_steps: Any
    starts: Any
    unique_starts: Any
    ended: Any
    error_sum: Any
    normal_episodes: Any


class ClusterFreePractice(GymnaxWrapper):
    def __init__(self, base, num_envs=128, mode='baseline', practice_envs=13,
                 capacity=2048, horizon=30, warmup=1007616, max_reuse=30, ema=.2):
        super().__init__(base)
        if mode not in ('baseline', 'uniform', 'prioritized') or not 0 <= practice_envs < num_envs:
            raise ValueError('Invalid cluster-free mode or lane count')
        if capacity < num_envs or min(capacity, horizon, max_reuse) < 1 or warmup < 0 or not 0 < ema <= 1:
            raise ValueError('Invalid cluster-free bank/horizon/reuse/EMA')
        self.num_envs, self.mode, self.capacity = num_envs, mode, capacity
        self.horizon, self.warmup, self.max_reuse, self.ema = horizon, warmup, max_reuse, ema
        self.practice_envs = practice_envs if mode != 'baseline' else 0
        self.assigned = jnp.arange(num_envs) >= num_envs-self.practice_envs
        self.step_fn = jax.vmap(base.step, in_axes=(0, 0, 0, None))
        self.reset_fn = jax.vmap(base.reset, in_axes=(0, None))
        self.obs_fn = jax.vmap(base.get_obs)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        obs, raw = self.reset_fn(jax.random.split(key, self.num_envs), params)
        history = jax.tree_util.tree_map(lambda x: jnp.zeros((5,)+x.shape, x.dtype), raw)
        bank = jax.tree_util.tree_map(lambda x: jnp.zeros((self.capacity,)+x.shape[1:], x.dtype), raw)
        zeros = jnp.zeros(self.num_envs, jnp.int32)
        info = {k: jnp.zeros(self.capacity, jnp.int32) for k in
                ('generation', 'reuse', 'updates', 'source_worker', 'source_episode', 'source_step')}
        info['score'] = jnp.ones(self.capacity)
        state = ClusterFreeState(raw, history, raw, zeros, zeros, jnp.asarray(0), bank, info,
            jnp.asarray(0), jnp.full(self.num_envs, -1, jnp.int32), zeros, zeros, zeros,
            jnp.zeros(self.num_envs), zeros, jnp.zeros(self.num_envs), jnp.ones(self.num_envs),
            jnp.zeros(self.num_envs), jnp.asarray(0), jnp.asarray(0), jnp.asarray(0), jnp.asarray(0),
            jnp.asarray(0), jnp.asarray(0.), jnp.asarray(0))
        return obs, state

    def _record(self, state, successor, reward, terminal, key):
        history = jax.tree_util.tree_map(lambda h, x: h.at[state.cursor].set(x), state.history, state.raw)
        before = jax.tree_util.tree_map(lambda h: h[(state.cursor-4) % 5], history)
        # Only the scalar achievement-reward component; flags are NOT read.
        proxy = reward - .1 * (successor.player_health-state.raw.player_health)
        event = (proxy > .5) & (state.entry < 0) & (state.history_count+1 >= 5)
        counts = state.candidate_count + event.astype(jnp.int32)
        selected = event & (jax.random.uniform(key, (self.num_envs,)) < 1/jnp.maximum(counts, 1))
        candidate = choose(selected, before, state.candidate)
        normal_done = terminal & (state.entry < 0)

        eligible = normal_done & (counts > 0)
        # Same worker-order FIFO insertion as a sequential loop, but one scatter
        # per leaf. Capacity >= lane count guarantees distinct valid slots.
        def insert(unused):
            serial = state.bank_total+jnp.cumsum(eligible.astype(jnp.int32))
            slots = jnp.where(eligible, (serial-1) % self.capacity, self.capacity)
            bank = jax.tree_util.tree_map(lambda a, x: a.at[slots].set(x, mode='drop'), state.bank, candidate)
            values = dict(generation=serial, reuse=jnp.zeros(self.num_envs, jnp.int32),
                updates=jnp.zeros(self.num_envs, jnp.int32), score=jnp.ones(self.num_envs),
                source_worker=jnp.arange(self.num_envs), source_episode=state.episode_serial,
                source_step=candidate.timestep)
            info = {k: a.at[slots].set(values[k], mode='drop') for k, a in state.bank_info.items()}
            return bank, info, state.bank_total+eligible.sum()
        bank, info, total = jax.lax.cond(eligible.any(), insert,
            lambda _: (state.bank, state.bank_info, state.bank_total), operand=None)
        return state.replace(history=history, candidate=candidate, candidate_count=counts,
            bank=bank, bank_info=info, bank_total=total,
            normal_episodes=state.normal_episodes+normal_done.sum())

    def _starts(self, key, normal, state, done):
        entries = jnp.full(self.num_envs, -1, jnp.int32)
        generations = jnp.zeros(self.num_envs, jnp.int32)
        def lane(i, carry):
            starts, entries, generations, state = carry
            usable = (state.bank_info['generation'] > 0) & (state.bank_info['reuse'] < self.max_reuse)
            def use(carry):
                starts, entries, generations, state = carry
                tie_key, sample_key = jax.random.split(jax.random.fold_in(key, i))
                probability = sampling_probabilities(state.bank_info['score'], usable,
                    self.mode == 'prioritized', tie_key)
                entry = jax.random.categorical(sample_key, jnp.where(usable, jnp.log(probability), -jnp.inf))
                starts = jax.tree_util.tree_map(lambda x, a: x.at[i].set(a[entry]), starts, state.bank)
                first = state.bank_info['reuse'][entry] == 0
                info = dict(state.bank_info)
                info['reuse'] = info['reuse'].at[entry].add(1)
                state = state.replace(bank_info=info, starts=state.starts+1,
                                      unique_starts=state.unique_starts+first.astype(jnp.int32))
                return starts, entries.at[i].set(entry), generations.at[i].set(info['generation'][entry]), state
            eligible = done[i] & self.assigned[i] & (state.total_steps >= self.warmup) & usable.any()
            return jax.lax.cond(eligible, use, lambda c: c, carry)
        initial = (normal, entries, generations, state)
        if not self.practice_envs:
            return initial
        eligible = (done & self.assigned).any() & (state.total_steps >= self.warmup)
        return jax.lax.cond(eligible, lambda c: jax.lax.fori_loop(
            self.num_envs-self.practice_envs, self.num_envs, lane, c), lambda c: c, initial)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, key, state, actions, params=None):
        sk, rk, bk, ck = jax.random.split(key, 4)
        obs, successor, reward, terminal, info = self.step_fn(
            jax.random.split(sk, self.num_envs), state.raw, actions, params)
        practice = state.entry >= 0
        elapsed = state.elapsed+1
        truncated = practice & (elapsed >= self.horizon) & ~terminal
        done = terminal | truncated
        state = self._record(state, successor, reward, terminal, ck)
        returns, lengths = state.episode_return+reward, state.episode_length+1
        info.update(returned_episode=terminal & ~practice,
            returned_episode_returns=jnp.where(terminal & ~practice, returns, 0.),
            returned_episode_lengths=jnp.where(terminal & ~practice, lengths, 0), timestep=successor.timestep)
        info.update({'cf/is_practice': practice, 'cf/truncated': truncated,
            'cf/ended': practice & done, 'cf/entry': state.entry, 'cf/generation': state.generation,
            'cf/elapsed': elapsed, 'cf/raw_reward': reward, 'cf/bootstrap_obs': obs})
        _, normal = self.reset_fn(jax.random.split(rk, self.num_envs), params)
        starts, entries, generations, state = self._starts(bk, normal, state, done)
        raw = choose(done, starts, successor)
        state = state.replace(raw=raw, cursor=(state.cursor+1) % 5,
            candidate_count=jnp.where(done, 0, state.candidate_count),
            history_count=jnp.where(done, 0, jnp.minimum(5, state.history_count+1)),
            entry=jnp.where(done, entries, state.entry), generation=jnp.where(done, generations, state.generation),
            elapsed=jnp.where(done, 0, elapsed), episode_return=jnp.where(done, 0., returns),
            episode_length=jnp.where(done, 0, lengths),
            episode_serial=state.episode_serial+done.astype(jnp.int32),
            total_steps=state.total_steps+self.num_envs, practice_steps=state.practice_steps+practice.sum())
        return self.obs_fn(raw), state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def record_prediction(self, state, info, value, successor_value, gamma):
        if not self.practice_envs:
            return state, jnp.zeros(self.num_envs)
        practice, ended = info['cf/is_practice'], info['cf/ended']
        first = info['cf/elapsed'] == 1
        start_value = jnp.where(first, value, state.score_start_value)
        accumulated = jnp.where(first, 0., state.score_return)
        discount = jnp.where(first, 1., state.score_discount)
        accumulated += discount * info['cf/raw_reward']
        discount *= gamma
        target = accumulated + discount * successor_value * info['cf/truncated']
        error = jax.lax.stop_gradient(jnp.abs(target-start_value))
        def lane(i, bank_info):
            entry = jnp.maximum(info['cf/entry'][i], 0)
            valid = ended[i] & (bank_info['generation'][entry] == info['cf/generation'][i])
            def update(bank_info):
                old = bank_info['score'][entry]
                score = jnp.where(bank_info['updates'][entry] == 0, error[i],
                                  (1-self.ema)*old+self.ema*error[i])
                return {**bank_info, 'score': bank_info['score'].at[entry].set(score),
                        'updates': bank_info['updates'].at[entry].add(1)}
            return jax.lax.cond(valid, update, lambda x: x, bank_info)
        bank_info = jax.lax.cond(ended.any(), lambda x: jax.lax.fori_loop(
            self.num_envs-self.practice_envs, self.num_envs, lane, x), lambda x: x, state.bank_info)
        state = state.replace(bank_info=bank_info,
            score_return=jnp.where(practice & ~ended, accumulated, 0.),
            score_discount=jnp.where(practice & ~ended, discount, 1.),
            score_start_value=jnp.where(practice & ~ended, start_value, 0.),
            ended=state.ended+ended.sum(), error_sum=state.error_sum+(error*ended).sum())
        return state, jnp.where(ended, error, 0.)


def save_cluster_free_archive(state, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = jax.device_get(dict(states=state.bank, **state.bank_info))
    with gzip.open(directory/'state_bank.pkl.gz', 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    summary = dict(total_steps=int(state.total_steps), practice_steps=int(state.practice_steps),
        practice_fraction=float(state.practice_steps/jnp.maximum(state.total_steps, 1)),
        normal_episodes=int(state.normal_episodes), inserted_snapshots=int(state.bank_total),
        practice_starts=int(state.starts), unique_snapshot_generations_used=int(state.unique_starts),
        ended_practices=int(state.ended), mean_absolute_return_prediction_error=float(state.error_sum/jnp.maximum(state.ended, 1)),
        current_bank_size=int((data['generation'] > 0).sum()), max_current_reuse=int(data['reuse'].max()),
        current_scored_snapshots=int((data['updates'] > 0).sum()),
        note='Trusted local pickle; snapshots retain factual simulator state, including flags, but IDs/flags never select or prioritize starts.')
    (directory/'bank_summary.json').write_text(json.dumps(summary, indent=2)+'\n')

"""Opt-in short-practice bootstrapping and trusted own-run archive continuation."""
import gzip
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def bootstrap_reward(reward, post_value, info, gamma, success_bootstrap=False):
    """Use POST-action/pre-reset values, never reset-state values or true terminals.

    This is a learning-target correction only. Keep environment/episode rewards
    unchanged, and still mark the reset done so GAE cannot cross into a new drill.
    """
    artificial = info['practice/truncated']
    if success_bootstrap:
        artificial = artificial | (info['practice/success'] & info['practice/ended'])
    mask = artificial & ~info['practice/environment_terminal']
    correction = gamma * jax.lax.stop_gradient(post_value) * mask
    return reward + correction, mask, correction


def read_archive(directory, capacity, window, max_reuse):
    """Only load trusted pickle files created by this project's completed runs."""
    from id_practice import TASKS, OFFSETS
    directory = Path(directory)
    summary = json.loads((directory/'archive_summary.json').read_text())
    if summary['tasks'] != list(TASKS) or summary['offsets'] != list(OFFSETS):
        raise ValueError('Continuation requires an ordinary 1/5/20 rare-four bank')
    with gzip.open(directory/'state_archive.pkl.gz','rb') as f:
        bank = pickle.load(f)
    with gzip.open(directory/'rarity_and_reuse.pkl.gz','rb') as f:
        rarity = pickle.load(f)
    counts = np.asarray(bank['counts'])
    if counts.shape != (22,) or np.any(counts<0) or np.any(counts>capacity):
        raise ValueError('Invalid archive counts')
    for name in ('lags','workers','episodes','event_steps'):
        if np.asarray(bank[name]).shape != (22,capacity):
            raise ValueError(f'Wrong archive shape: {name}')
    if np.any(np.asarray(bank['totals'])<counts):
        raise ValueError('Archive totals smaller than counts')
    if np.asarray(rarity['normal_window']).shape != (window,22):
        raise ValueError('Rarity window mismatch')
    if not np.array_equal(np.asarray(rarity['normal_sum']),np.asarray(rarity['normal_window']).sum(axis=0)):
        raise ValueError('Inconsistent rarity sums')
    reuse = np.asarray(rarity['reuse_counts'])
    if reuse.shape != (22,capacity) or np.any(reuse<0) or np.any(reuse>max_reuse):
        raise ValueError('Invalid reuse counters; quotas will not be reset')
    flags = np.asarray(bank['states'].achievements)
    for i,count in enumerate(counts):
        if flags[i,:int(count),i].any():
            raise ValueError('Target achievement already set in an archived start')
        if not np.isin(np.asarray(bank['lags'])[i,:int(count)],OFFSETS).all():
            raise ValueError('Invalid snapshot offset')
    return bank,rarity


def seed_archive(state, payload):
    """Restore banks/quotas/rarity; start fresh live episodes and phase counters."""
    bank,rarity = payload
    def compatible(template,saved):
        if template.shape != saved.shape or template.dtype != saved.dtype:
            raise ValueError('Saved archive is incompatible with this environment')
        return jnp.asarray(saved)
    archive = jax.tree_util.tree_map(compatible,state.archive,bank['states'])
    # Avoid source-episode provenance collisions with the previous training phase.
    next_serial = int(np.asarray(bank['episodes']).max())+1
    return state.replace(archive=archive,archive_counts=jnp.asarray(bank['counts']),
        archive_totals=jnp.asarray(bank['totals']),archive_lags=jnp.asarray(bank['lags']),
        archive_workers=jnp.asarray(bank['workers']),archive_episodes=jnp.asarray(bank['episodes']),
        archive_event_steps=jnp.asarray(bank['event_steps']),
        episode_serial=jnp.full_like(state.episode_serial,next_serial),
        reuse_counts=jnp.asarray(rarity['reuse_counts']),rarity_window=jnp.asarray(rarity['normal_window']),
        rarity_sum=jnp.asarray(rarity['normal_sum']),rarity_total=jnp.asarray(rarity['normal_episodes']),
        rarity_last_update=jnp.asarray(rarity['normal_episodes']),rarity_selected=jnp.asarray(rarity['selected']))

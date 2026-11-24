"""Read the losslessly encoded local observation and spectator state at frame t."""
from types import SimpleNamespace

import numpy as np


def observation(episode, t):
    bits = np.unpackbits(episode['local_bits'][t], axis=-1, count=1323)
    return np.concatenate((bits, episode['local_stats'][t]), axis=-1)


def terrain(episode, t):
    count = len(episode['player_position'])
    if not -count <= t < count:
        raise IndexError(f'World frame {t} is outside an episode of {count} frames')
    t %= count
    grid = episode['initial_map'].copy()
    n = np.searchsorted(episode['change_step'], t, side='right')
    # Changes within one frame target distinct cells, but later frames can revisit them.
    for cell, tile in zip(episode['change_cell'][:n], episode['change_tile'][:n]):
        grid.flat[cell] = tile
    return grid


def spectator_state(episode, t):
    fields = dict(map=terrain(episode, t), achievements=episode['achievements'][t],
                  player_position=episode['player_position'][t])
    for group in ('zombies', 'cows', 'skeletons', 'arrows'):
        fields[group] = SimpleNamespace(position=episode[group + '_position'][t],
                                        mask=episode[group + '_mask'][t])
    return SimpleNamespace(**fields)

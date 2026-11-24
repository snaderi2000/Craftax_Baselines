"""Export the installed Craftax Classic generator's intermediate maps for an archived reset."""
import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.envs.craftax_state import EnvParams, StaticEnvParams
from craftax.craftax_classic.world_gen import generate_world


def capture_world(reset_key):
    """Observe the generator's immutable arrays as each assignment takes effect."""
    variables = ('map', 'water', 'mountain', 'path_x', 'tree_noise')
    snapshots = {name: [] for name in variables}

    def capture(frame, event, arg):
        if event in ('line', 'return'):
            state = frame.f_locals
            for name, history in snapshots.items():
                if name in state and (not history or history[-1] is not state[name]):
                    history.append(state[name])
        return capture

    def on_call(frame, event, arg):
        return capture if frame.f_code is generate_world.__code__ else None

    old_trace = sys.gettrace()
    try:
        sys.settrace(on_call)
        world = generate_world(jnp.asarray(reset_key), EnvParams(), StaticEnvParams())
    finally:
        sys.settrace(old_trace)
    return snapshots, world


def build(archive):
    with np.load(archive) as ep:
        key, episode, expected = ep['reset_key'], ep['episode_id'].item(), ep['initial_map']
    history, state = capture_world(key)
    np.testing.assert_array_equal(np.asarray(state.map), expected,
                                  err_msg='Regenerated map differs from the archived reset.')
    # v1.5 assigns the map twice for the two path directions; expose them as one step.
    names = ('water', 'sand', 'mountain', 'pathX', 'paths', 'caves', 'coal', 'iron',
             'diamond', 'trees', 'lava', 'spawn', 'guarantee')
    if len(history['map']) != len(names) or len(history['water']) != 2:
        raise ValueError('Unexpected intermediate fields in this Craftax version')
    fields = dict(waterRaw=history['water'][0], water=history['water'][1],
                  mountain=history['mountain'][-1], pathX=history['path_x'][0],
                  tree=history['tree_noise'][0])
    stages = []
    previous = np.full(expected.shape, -1)
    for name, grid in zip(names, history['map']):
        if name == 'pathX':
            continue
        grid = np.asarray(grid)
        changed = grid != previous
        stages.append(dict(id=name, map=grid.ravel().tolist(), changed=int(changed.sum())))
        if name == 'guarantee':
            diamond, = np.argwhere(changed)
        previous = grid
    return dict(episode=episode, size=expected.shape[0], spawn=np.asarray(state.player_position).tolist(),
                resetKey=key.tolist(), matchedTiles=expected.size, craftaxVersion=version('craftax'),
                fields={name: np.asarray(value).ravel().tolist() for name, value in fields.items()},
                stages=stages, diamond=diamond.tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    data = build(args.episode)
    args.output.write_text('window.WORLDGEN=' + json.dumps(data, separators=(',', ':')) + ';\n')
    print(data['episode'], data['resetKey'], data['matchedTiles'], data['diamond'])
    print([(stage['id'], stage['changed']) for stage in data['stages']])

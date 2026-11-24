"""Train pin-following heads on frozen V5 features and test actual arrival in Classic."""
import argparse
import csv
from collections import deque
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from evaluate_reward_shaped_ppo import jax, jnp, make_craftax_env_from_name
import flax.linen as nn
from flax import serialization
import optax
from cap_reward_shaped_runs import try_evaluation_lock
from concept_mapping.probe_v5_water import model, OUT as WATER_PROBE
from models.actor_critic_map_mlp import GlobalMapReader, map_mlp_network
from symbolic_map_memory import LOCAL_SIZE, with_map_memory

MOVES = np.array([(0, -1), (0, 1), (-1, 0), (1, 0)])
BITS = np.array([1, 2, 4, 8])
CAN_WALK = (3, 7, 8, 14)  # Stored terrain IDs for grass, wood, path, sand.
MAX_ROUTE = 32
HORIZON = 96
PREFIX = 256
SEED = 231126731
OUT = ROOT / 'concept_mapping/runs/v5_supervised_pin_navigation'


def routes(terrain, position):
    """Label every candidate with distance and ALL shortest first moves through known terrain."""
    walk = np.isin(terrain, CAN_WALK)
    distances = np.full(walk.shape, -1, np.int16)
    first = np.zeros(walk.shape, np.uint8)
    origin = tuple(position)
    distances[origin] = 0
    queue = deque([origin])
    directions = MOVES.tolist()
    while queue:
        r, c = queue.popleft()
        depth = distances[r, c]
        if depth >= MAX_ROUTE:
            continue
        for i, (dr, dc) in enumerate(directions):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < 64 and 0 <= nc < 64 and walk[nr, nc]):
                continue
            if distances[nr, nc] == -1:
                distances[nr, nc] = depth + 1
                queue.append((nr, nc))
            if distances[nr, nc] == depth + 1:
                first[nr, nc] |= (1 << i) if depth == 0 else first[r, c]
    targets = np.argwhere(distances > 0)
    delta = targets - position
    manhattan = np.abs(delta).sum(axis=1)
    distance = distances[tuple(targets.T)]
    labels = (first[tuple(targets.T)][:, None] & BITS) != 0
    decreases = np.abs(delta[:, None] - MOVES).sum(axis=-1) < manhattan[:, None]
    return targets, labels, distance, manhattan, ~(decreases & labels).any(axis=1)


def training_set(snapshots, rng):
    ids, pins, answers = [], [], []
    for i, (terrain, pos) in enumerate(zip(snapshots['memory'], snapshots['pos'])):
        targets, labels, distance, direct, _ = routes(terrain, pos)
        for mask in (distance <= 5, (distance > 5) & (distance == direct), distance > direct):
            choices = np.flatnonzero(mask)
            if len(choices):
                sample = rng.choice(choices, min(32, len(choices)), replace=False)
                ids.extend([i] * len(sample))
                pins.extend(targets[sample])
                answers.extend(labels[sample])
    return np.asarray(ids), np.asarray(pins), np.asarray(answers, np.float32)


class Worker(nn.Module):
    use_map: bool

    @nn.compact
    def __call__(self, live, regions, pos, pin):
        # Give both heads the same live view, position and destination.
        local = live[:, :1323].reshape(-1, 7, 9, 21)
        ground = local[..., [2, 6, 7, 13]].sum(-1)
        mobs = local[..., 17:].max(-1)
        delta = pin - pos
        direction = delta / jnp.maximum(1, jnp.abs(delta).sum(-1, keepdims=True))
        geo = jnp.concatenate([direction, delta / 32, pos / 63, (pos % 8) / 7,
                               (pin % 8) / 7], axis=-1)
        features = [ground.reshape(-1, 63), mobs.reshape(-1, 63), geo]
        if self.use_map:
            # Preserve V5's full-board map. Add region positions relative to both
            # player and pin before pooling, so the worker can locate either one.
            coords = jnp.indices((8, 8)).transpose(1, 2, 0) * 8 + 3.5
            rel = (coords - pos[:, None, None, :]) / 32
            target_rel = (coords - pin[:, None, None, :]) / 32
            tokens = jnp.concatenate((regions, rel, target_rel), -1)
            tokens = nn.relu(nn.Dense(16)(tokens))
            features.append(nn.relu(nn.Dense(64)(tokens.reshape(len(pos), -1))))
        hidden = nn.relu(nn.Dense(128)(jnp.concatenate(features, -1)))
        hidden = nn.relu(nn.Dense(128)(hidden))
        return nn.Dense(4)(hidden)


def train_workers(out):
    rng = np.random.default_rng(SEED)
    with np.load(WATER_PROBE / 'snapshots.npz') as archive:
        snapshots = {k: archive[k] for k in ('memory', 'pos', 'live', 'regions')}
        source = str(archive['checkpoint'])
    ids, pins, labels = training_set(snapshots, rng)
    # The frozen archive has one snapshot per world. Use 768 for fitting.
    train = np.flatnonzero(ids < 768)
    valid = np.flatnonzero(ids >= 768)
    report = {'checkpoint': source, 'training_worlds': int(len(np.unique(ids[train]))),
              'training_labels': len(train), 'validation_labels': len(valid), 'offline': {}}
    arrays = [jnp.asarray(snapshots[k]) for k in ('live', 'regions', 'pos')]
    ids, pins, labels = map(jnp.asarray, (ids, pins, labels))
    def batch(index):
        return [a[ids[index]] for a in arrays] + [pins[index]]
    for mode in ('local', 'map'):
        net = Worker(mode == 'map')
        optimizer = optax.adam(0.001)
        weights = net.init(jax.random.PRNGKey(SEED), *batch(train[:1]))
        state = optimizer.init(weights)

        @jax.jit
        def update(weights, state, index):
            def loss(params):
                y = labels[index]
                logp = jax.nn.log_softmax(net.apply(params, *batch(index)))
                return -((y * logp).sum(-1) / y.sum(-1)).mean()
            grads = jax.grad(loss)(weights)
            updates, state = optimizer.update(grads, state, weights)
            return optax.apply_updates(weights, updates), state

        @jax.jit
        def hits(params, index):
            actions = net.apply(params, *batch(index)).argmax(-1)
            return labels[index, actions].sum()

        # Identical sampled minibatches and update counts for both workers.
        train_rng = np.random.default_rng(SEED + 1)
        for _ in range(2000):
            ix = train_rng.choice(train, size=256)
            weights, state = update(weights, state, ix)
        path = out / (mode + '.msgpack')
        path.write_bytes(serialization.to_bytes(weights))
        saved = serialization.from_bytes(weights, path.read_bytes())
        n = sum(int(hits(saved, valid[i:i+256])) for i in range(0, len(valid), 256))
        report['offline'][mode] = n / len(valid)
        print(mode, 'held-out first move', n, '/', len(valid), flush=True)
    (out / 'training.json').write_text(json.dumps(report, indent=2) + '\n')


def selected_pin(terrain, pos, rng, prefer_detour):
    targets, _, distance, direct, away = routes(terrain, pos)
    delta = abs(targets - pos)
    offscreen = (delta[:, 0] > 3) | (delta[:, 1] > 4)
    valid = offscreen & (distance >= 6)
    hard = valid & away
    pool = np.flatnonzero(hard if prefer_detour and hard.any() else valid)
    if not len(pool):
        return None
    i = rng.choice(pool)
    return targets[i], int(distance[i]), int(direct[i]), bool(away[i])


def live_evaluation(out, world_count, batch_size=64):
    cfg, path, weights = model()
    assert str(path) == json.loads((out/'training.json').read_text())['checkpoint']
    base = map_mlp_network(cfg)
    reader = GlobalMapReader(cfg['LAYER_SIZE'], features_only=True, spatial_regions=True)
    reader_weights = {'params': weights['params']['global_reader']}
    env = with_map_memory(make_craftax_env_from_name(cfg['ENV_NAME'], False), cfg)
    default = env.default_params
    modes = ('local', 'map', 'base')
    heads = {k: Worker(k == 'map') for k in modes if k != 'base'}
    head_weights = {k: serialization.msgpack_restore((out / (k+'.msgpack')).read_bytes()) for k in heads}
    reset = jax.vmap(env.reset, in_axes=(0, None))
    step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    def keys(worlds, n):
        return jax.vmap(lambda w: jax.random.fold_in(jax.random.PRNGKey(n), w))(worlds)
    def tick_keys(original, t):
        return jax.vmap(lambda k: jax.random.fold_in(k, t))(original)
    choose = jax.vmap(jax.random.categorical)

    @jax.jit
    def warmup(worlds):
        obs, state = reset(jax.vmap(jax.random.PRNGKey)(worlds), default)
        ak, dk = keys(worlds, 401), keys(worlds, 402)
        def one(carry, t):
            obs, state, done = carry
            policy, _ = base.apply(weights, obs)
            action = choose(tick_keys(ak, t), policy.logits)
            obs, state, _, died, _ = step(tick_keys(dk, t), state, action, default)
            return (obs, state, done | died), None
        return jax.lax.scan(one, (obs, state, jnp.zeros(len(worlds), bool)), jnp.arange(PREFIX))[0]

    def build_rollout(mode):
        @jax.jit
        def rollout(obs, state, worlds, pins):
            ak, dk = keys(worlds, 501), keys(worlds, 502)
            def one(carry, t):
                obs, state, finished, hit, death, length = carry
                if mode == 'base':
                    policy, _ = base.apply(weights, obs)
                    action = choose(tick_keys(ak, t), policy.logits)
                else:
                    regions = reader.apply(reader_weights, obs[:, LOCAL_SIZE:]) if mode == 'map' else None
                    logits = heads[mode].apply(head_weights[mode], obs[:, :LOCAL_SIZE], regions, state.position, pins)
                    action = choose(tick_keys(ak, t), logits / 0.75) + 1
                new_obs, new_state, _, terminal, _ = step(tick_keys(dk, t), state, action, default)
                arrived = (new_state.position == pins).all(-1) & ~terminal & ~finished
                active = ~finished
                return (new_obs, new_state, finished | terminal | arrived, hit | arrived,
                        death | (active & terminal), length + active), new_state.position
            zeros = jnp.zeros(len(worlds), bool)
            start = (obs, state, zeros, zeros, zeros, jnp.zeros(len(worlds), jnp.int32))
            (_, _, _, hit, death, length), trace = jax.lax.scan(one, start, jnp.arange(HORIZON))
            return hit, death, length, trace
        return rollout

    runners = {k: build_rollout(k) for k in modes}
    rng = np.random.default_rng(SEED + 2)
    worlds = rng.integers(0, 2**32, world_count, dtype=np.uint32)
    assert len(set(worlds)) == len(worlds)
    rows, traces, screened = [], [], {'dead_before_pin': 0, 'no_eligible_pin': 0}
    for offset in range(0, world_count, batch_size):
        batch = worlds[offset:offset+batch_size]
        obs, state, dead = warmup(jnp.array(batch))
        maps, positions, dead = jax.device_get((state.terrain, state.position, dead))
        pins = positions.copy()
        selected = {}
        for i, w in enumerate(batch):
            if dead[i]:
                screened['dead_before_pin'] += 1
                continue
            pick = selected_pin(maps[i], positions[i], np.random.default_rng(int(w)), (i+offset)%2 == 0)
            if pick is None:
                screened['no_eligible_pin'] += 1
                continue
            pins[i], dist, direct, away = pick
            selected[i] = {'seed': int(w), 'start_row': int(positions[i, 0]), 'start_col': int(positions[i, 1]),
                           'pin_row': int(pins[i, 0]), 'pin_col': int(pins[i, 1]), 'shortest_route': dist,
                           'manhattan': direct, 'first_move_away': away}
        for mode, run in runners.items():
            hit, death, length, trace = jax.device_get(run(obs, state, jnp.array(batch), jnp.asarray(pins)))
            for i, item in selected.items():
                rows.append(dict(item, worker=mode, reached=int(hit[i]), died=int(death[i]), steps=int(length[i])))
                if mode == 'map':
                    traces.append((item, int(hit[i]), maps[i], np.vstack([positions[i], trace[:length[i], i]])))
        print('Worlds screened', offset+len(batch), '/', world_count, flush=True)
    if not rows:
        raise RuntimeError('No eligible pins.')
    with (out / 'episodes.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    stats = {}
    for mode in modes:
        own = [row for row in rows if row['worker'] == mode]
        stats[mode] = {}
        for label, group in [('all', own), ('first_move_away', [r for r in own if r['first_move_away']]),
                             ('longer_than_manhattan', [r for r in own if r['shortest_route'] > r['manhattan']])]:
            stats[mode][label] = dict(worlds=len(group), reached=sum(r['reached'] for r in group),
                                      died=sum(r['died'] for r in group))
    paired = {}
    for control in ('local', 'base'):
        other = {r['seed']: r for r in rows if r['worker'] == control}
        diff = np.array([r['reached'] - other[r['seed']]['reached'] for r in rows if r['worker'] == 'map'])
        boots = np.random.default_rng(SEED+3).choice(diff, (10000, len(diff))).mean(-1)
        paired[control] = dict(map_only=int((diff == 1).sum()), control_only=int((diff == -1).sum()),
                               difference_pp=100*diff.mean(), ci95_pp=(100*np.quantile(boots, [.025,.975])).tolist())
    summary = dict(screened_worlds=world_count, excluded=screened, prefix=PREFIX, horizon=HORIZON,
                   pin_distance=[6, MAX_ROUTE], workers=stats, paired_vs_map=paired,
                   first_move_away_definition='every shortest first step increases Manhattan distance')
    (out/'results.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)
    plot_paths(out, traces)


def plot_paths(out, traces):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig, axes = plt.subplots(1, 2, figsize=(8.5,4.4), constrained_layout=True)
    for ax, arrived in zip(axes, (1, 0)):
        options = [v for v in traces if v[1] == arrived]
        if not options:
            ax.axis('off')
            continue
        item, _, terrain, path = next((v for v in options if v[0]['first_move_away']), options[0])
        board = np.where(terrain == 0, 0, np.where(np.isin(terrain,CAN_WALK), 1, 2))
        ax.imshow(board, cmap=ListedColormap(['#dfe1e1','#f6f4e9','#738779']), vmin=0, vmax=2)
        ax.plot(path[:,1],path[:,0], color='#087392', lw=2)
        ax.scatter(path[0,1],path[0,0],c='white',edgecolor='#087392',s=50,zorder=3,label='Start')
        ax.scatter(item['pin_col'],item['pin_row'],marker='*',c='#bc462f',s=150,zorder=4,label='Assigned pin')
        rr=np.r_[path[:,0],item['pin_row']]; cc=np.r_[path[:,1],item['pin_col']]
        ax.set_xlim(max(-.5,cc.min()-4),min(63.5,cc.max()+4))
        ax.set_ylim(min(63.5,rr.max()+4),max(-.5,rr.min()-4))
        ax.set_title(('Reached' if arrived else 'Missed') + f" · world {item['seed']}",fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc='best',fontsize=8)
    fig.suptitle('Actual routes to externally assigned pins\nV5 map worker on held-out Craftax worlds')
    fig.savefig(out/'routes.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--worlds', type=int, default=256)
    args = parser.parse_args()
    if str(args.gpu) != os.environ.get('CUDA_VISIBLE_DEVICES'):
        parser.error('--gpu must match CUDA_VISIBLE_DEVICES')
    lock = try_evaluation_lock(args.gpu)
    if lock is None:
        raise RuntimeError('GPU is reserved.')
    with lock:
        if jax.default_backend() != 'gpu':
            raise RuntimeError('Expected a GPU.')
        args.out.mkdir(parents=True, exist_ok=True)
        if not (args.out/'training.json').is_file():
            train_workers(args.out)
        if (args.out/'results.json').exists():
            raise FileExistsError('This evaluation already finished.')
        live_evaluation(args.out, args.worlds)

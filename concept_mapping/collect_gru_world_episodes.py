"""Archive all transitions and spectator frames from an existing local GRU evaluation."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from concept_mapping.world_episode import observation, terrain


def archive_fields(obs, raw):
    import jax.numpy as jnp
    # The local map has 63 cells with 17 one-hot terrain and four binary mob channels.
    fields = dict(map=raw.map.astype(jnp.uint8), player_position=raw.player_position.astype(jnp.int16),
                  local_bits=jnp.packbits(obs[:, :1323].astype(jnp.uint8), axis=-1),
                  local_stats=obs[:, 1323:], achievements=raw.achievements,
                  arrow_directions=raw.arrow_directions.astype(jnp.int8))
    for group in ('zombies', 'cows', 'skeletons', 'arrows'):
        mobs = getattr(raw, group)
        fields[group + '_position'] = mobs.position.astype(jnp.int16)
        fields[group + '_mask'] = mobs.mask
    return fields


def write_json(path, data):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def save_episode(path, data):
    tmp = path.with_suffix('.npz.tmp')
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for key, value in data.items():
            with archive.open(key + '.npy', 'w') as f:
                np.lib.format.write_array(f, np.ascontiguousarray(value), allow_pickle=False)
    tmp.replace(path)


def encode_lane(frames, transitions, lane, length):
    data = {k: v[:length, lane] for k, v in transitions.items()}
    data.update({k: v[:length+1, lane] for k, v in frames.items() if k != 'map'})
    maps = np.ascontiguousarray(frames['map'][:length+1, lane])
    flat = maps.reshape(length+1, -1)
    steps, cells = np.nonzero(flat[1:] != flat[:-1])
    data.update(initial_map=maps[0], change_step=(steps+1).astype(np.uint16),
                change_cell=cells.astype(np.uint16), change_tile=flat[steps+1, cells])
    assert data['dones'][-1] and not data['dones'][:-1].any()
    for t in (0, length//2, length):
        np.testing.assert_array_equal(terrain(data, t), maps[t])
        row, col = data['player_position'][t]
        # The simulator can return a terminal coordinate one tile beyond an edge.
        padded = np.pad(maps[t], ((4, 4), (5, 5)), constant_values=1)
        local = observation(data, t)[:1323].reshape(7, 9, 21)
        np.testing.assert_array_equal(local[:, :, :17].argmax(-1), padded[row+1:row+8, col+1:col+10])
    return data


def worker(args, meta, original):
    gpu = args.gpus[args.worker]
    from cap_reward_shaped_runs import try_evaluation_lock
    from run_ppo_study import require_free
    from run_reward_shaped_architectures import environment
    lock = try_evaluation_lock(gpu)
    if lock is None:
        raise RuntimeError(f'GPU {gpu} reserved')
    require_free(gpu)
    os.environ.update(environment(gpu))
    from evaluate_reward_shaped_ppo import build_evaluator
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from craftax.craftax_env import make_craftax_env_from_name
    from craftax.craftax_classic.constants import Achievement
    from ppo_rnn_og import ActorCriticRNN
    cfg = json.loads((args.evaluation.parent/'inference_config.json').read_text())
    if not cfg.get('RECURRENT') or cfg.get('SYMBOLIC_ENCODER') != 'mlp' or cfg.get('MOTION_INPUTS') or cfg.get('WORLD_VIEW'):
        raise ValueError('Expected the regular local-view GRU')
    env = make_craftax_env_from_name(cfg['ENV_NAME'], auto_reset=False)
    net = ActorCriticRNN(env.action_space(env.default_params).n, cfg)
    path = Path(meta['checkpoint'])
    payload = path/'default' if (path/'default').is_dir() else path
    params = ocp.PyTreeCheckpointer().restore(str(payload))['params']
    batch = meta['batch']
    evaluate = build_evaluator(env, net, params, cfg, seed=meta['evaluation_seed'], batch=batch,
                               temperature=meta['temperature'], max_steps=meta['max_episode_steps'],
                               archive_fields=archive_fields)
    for start in range(args.worker*batch, args.episodes, len(args.gpus)*batch):
        index = args.out/'batches'/f'{start:04d}.json'
        if index.exists() and all((args.out/row['file']).is_file() for row in json.loads(index.read_text())):
            continue
        print(f'GPU {gpu} collecting batch {start}', flush=True)
        ids = jnp.arange(start, start+batch)
        result = jax.device_get(evaluate(ids))
        _, finished, returns, shaped, lengths, achievements, lava, death, timeout, (frames, actions) = result
        keys = jax.vmap(lambda i: jax.random.split(jax.random.fold_in(jax.random.PRNGKey(meta['evaluation_seed']), i)))(ids)
        keys = np.asarray(keys)
        rows = []
        for lane in range(min(batch, args.episodes-start)):
            eid = start+lane
            length = int(lengths[lane])
            data = encode_lane(frames, actions, lane, length)
            data.update(reset_key=keys[lane, 0], rollout_key=keys[lane, 1],
                        episode_id=np.int32(eid), checkpoint=np.int64(path.name))
            np.testing.assert_array_equal(data['achievements'][-1], achievements[lane])
            row = dict(episode=eid, length=length, original_return=float(returns[lane]),
                       shaped_return=float(shaped[lane]), terminated=bool(finished[lane]),
                       lava=bool(lava[lane]), nonlava_death=bool(death[lane]), timeout=bool(timeout[lane]),
                       all_22=bool(achievements[lane].all()))
            row.update({a.name.lower(): int(achievements[lane, a.value]) for a in Achievement})
            ref = original[eid]
            row['matches_original_evaluation'] = all(
                abs(float(ref[k])-v)<1e-5 if k.endswith('return') else str(v)==ref[k] for k,v in row.items())
            episode_file = f'episodes/episode_{eid:04d}.npz'
            save_episode(args.out/episode_file, data)
            row.update(file=episode_file, final_row=int(data['player_position'][-1,0]),
                       final_col=int(data['player_position'][-1,1]),
                       initial_map_sha256=hashlib.sha256(data['initial_map'].tobytes()).hexdigest())
            rows.append(row)
        write_json(index, rows)
        print(f'GPU {gpu} saved {len(rows)} episodes starting at {start}', flush=True)
        del frames, actions, result, data
    lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    parser.add_argument('--worker', type=int, default=None)
    args = parser.parse_args()
    ev = json.loads((args.evaluation/'summary.json').read_text())
    if ev.get('world_seeds_file') or not 0 < args.episodes <= ev['episodes'] or len(set(args.gpus)) != len(args.gpus):
        raise ValueError('Expected distinct GPUs and independent recorded evaluation episodes')
    config = json.loads((args.evaluation.parent/'experiment_metadata.json').read_text())
    batch = config['evaluation']['batch']
    with (args.evaluation/'episodes.csv').open() as f:
        original = {int(r['episode']): r for r in csv.DictReader(f)}
    meta = {k: ev[k] for k in ('checkpoint','training_steps','temperature','max_episode_steps','evaluation_seed','architecture')}
    meta.update(batch=batch, episodes=args.episodes, source_evaluation=str(args.evaluation.resolve()),
                format_version=1, coordinates='zero-based (row, column), origin at upper left',
                states='T+1 frames, with frame T terminal. actions[t], rewards[t], original_rewards[t], dones[t] and values[t] describe frame t to t+1.',
                local='Lossless packed 7x9x21 symbolic map plus the original 22 floating-point stats. Use concept_mapping.world_episode.observation.',
                world='initial_map plus chronological absolute tile replacements, all entity positions/masks, player_position and achievements at every frame. Use concept_mapping.world_episode.spectator_state.',
                occupancy='Use player_position[:-1] for one visit per action. The last position is the terminal location.',
                failure='all_22 measures achievement completion. timeout measures reaching the step limit alive. lava and nonlava_death record deaths independently of achievements.')
    if args.worker is not None:
        return worker(args, meta, original)
    for dirname in ('episodes','batches'):
        (args.out/dirname).mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.out/'.collection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = args.out/'dataset_metadata.json'
        if info.exists() and meta != json.loads(info.read_text()):
            raise ValueError('Existing dataset metadata differs')
        write_json(info, meta)
        needed = set()
        for first in range(0, args.episodes, batch):
            index = args.out/'batches'/f'{first:04d}.json'
            if not index.exists() or not all((args.out/r['file']).is_file() for r in json.loads(index.read_text())):
                needed.add(first//batch % len(args.gpus))
        workers = [subprocess.Popen([sys.executable, '-u', __file__, *sys.argv[1:], '--worker', str(i)])
                   for i in sorted(needed)]
        if any(code != 0 for code in [proc.wait() for proc in workers]):
            raise RuntimeError('A collection worker failed')
        rows = [r for file in sorted((args.out/'batches').glob('*.json')) for r in json.loads(file.read_text())]
        assert [r['episode'] for r in rows] == list(range(args.episodes))
        temp = args.out/'episodes.csv.tmp'
        with temp.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(args.out/'episodes.csv')
        summary = dict(episodes=len(rows), transitions=sum(r['length'] for r in rows),
                       unique_initial_maps=len({r['initial_map_sha256'] for r in rows}),
                       all_22=sum(r['all_22'] for r in rows), diamond=sum(r['collect_diamond'] for r in rows),
                       timeout=sum(r['timeout'] for r in rows), lava=sum(r['lava'] for r in rows),
                       nonlava_death=sum(r['nonlava_death'] for r in rows),
                       evaluation_mismatches=[r['episode'] for r in rows if not r['matches_original_evaluation']])
        write_json(args.out/'summary.json', summary)
        print(summary, flush=True)


if __name__ == '__main__':
    main()

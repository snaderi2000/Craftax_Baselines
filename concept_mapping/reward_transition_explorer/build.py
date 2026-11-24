"""Build an offline HTML explorer from the existing label-free k-means experiment.

Run with the repository's jax-v100-cuda12 Python. Only NumPy/SciPy/Pillow are
needed; building and serving never run a policy or mutate the source datasets.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from concept_mapping.test_achievement_clustering_offline import classify_clusters

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / 'concept_mapping/runs/achievement_clustering_offline_10b_batch2000'
DEFAULT_OUT = ROOT / 'concept_mapping/runs/reward_transition_explorer'
METHODS = ['critic_delta', 'actor_delta', 'raw_observation_delta',
           'random_actor_delta', 'action_inventory_delta']
EXTRA_NAMES = ['health_gain_only', 'health_loss_only', 'other_no_new_achievement',
               'multiple_new_achievements']


def normalize(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def eligible(rewards, mode):
    if mode == 'achievement_reward':
        return np.isclose(rewards, 1.0, rtol=0, atol=1e-6)
    raise ValueError('Unknown reward filter')


def validate_bank(bank):
    n = len(bank['rewards'])
    for name in ('before', 'after'):
        if bank[name].shape != (n, 1345) or not np.isfinite(bank[name]).all():
            raise ValueError(f'Invalid {name} observations; expected Classic [N,1345]')
    for name in ('labels', 'steps', 'actions', 'episode_uid', 'split_id'):
        if bank[name].shape != (n,):
            raise ValueError(f'Invalid {name}')
    if not np.isin(bank['split_id'], [0, 1]).all():
        raise ValueError('Invalid split IDs')
    if set(bank['episode_uid'][bank['split_id'] == 0]) & set(bank['episode_uid'][bank['split_id'] == 1]):
        raise ValueError('Reference and held-out episodes overlap')
    if not np.isfinite(bank['rewards']).all() or not np.isin(bank['labels'], np.arange(26)).all():
        raise ValueError('Invalid rewards or diagnostic labels')


def describe_state(obs):
    grid = obs[:1323].reshape(63, 21)
    # Compact categorical map and mob bitmask, decoded into the original view.
    return dict(blocks=grid[:, :17].argmax(1).tolist(),
                mobs=((grid[:, 17:] > .5) * np.array([1, 2, 4, 8])).sum(1).tolist(),
                stats=np.round(obs[1323:1339] * 10, 4).tolist(),
                direction=int(obs[1339:1343].argmax()),
                light=round(float(obs[1343]), 5), sleeping=bool(obs[1344] > .5))


def describe_rows(bank, indices):
    return [dict(id=int(i), episode=int(bank['episode_uid'][i]), step=int(bank['steps'][i]),
                 split=int(bank['split_id'][i]), label=int(bank['labels'][i]),
                 action=int(bank['actions'][i]), reward=float(bank['rewards'][i]),
                 before=describe_state(bank['before'][i]), after=describe_state(bank['after'][i]))
            for i in indices]


def diagnostics(bank, tr, te, a, b, k):
    metrics, precision, recall, f1, support, mapping, ref, test, predicted = classify_clusters(
        bank['labels'][tr], a, bank['labels'][te], b, k)
    # The upstream metric is undefined when the held-out set has no achievements.
    if not np.any(support[:22]):
        metrics['achievement_macro_f1'] = None
        metrics['achievement_macro_recall'] = None
    return dict(metrics=metrics, mapping=mapping.tolist(), reference_table=ref.T.tolist(),
                test_table=test.T.tolist(), classes=[dict(label=i, precision=float(precision[i]),
                    recall=float(recall[i]), f1=float(f1[i]), support=int(support[i])) for i in range(26)])


def assignment(x, centers):
    result = []
    cn = (centers ** 2).sum(1)
    for start in range(0, len(x), 4096):
        batch = x[start:start + 4096]
        result.append(((batch ** 2).sum(1)[:, None] + cn - 2 * batch @ centers.T).argmin(1))
    return np.concatenate(result)


def fit_kmeans(x, k, seed, iterations=30):
    """CPU Lloyd k-means with k-means++ and deterministic empty-cluster repair.

    Inputs are already L2-normalized. No rewards, labels or held-out features
    are inputs to initialization or fitting. Centers are Euclidean means.
    """
    x = np.asarray(x, np.float32)
    if not 2 <= k <= len(x) or not np.isfinite(x).all():
        raise ValueError('k must be between 2 and the number of reference points')
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))].copy()]
    nearest = ((x - centers[0]) ** 2).sum(1)
    for _ in range(1, k):
        total = float(nearest.sum(dtype=np.float64))
        ix = rng.choice(len(x), p=nearest.astype(np.float64) / total) if total > 1e-10 else rng.integers(len(x))
        centers.append(x[ix].copy())
        nearest = np.minimum(nearest, ((x - centers[-1]) ** 2).sum(1))
    centers = np.stack(centers)
    previous = None
    for _ in range(iterations):
        a = assignment(x, centers)
        if previous is not None and np.array_equal(a, previous):
            break
        previous = a
        counts = np.bincount(a, minlength=k)
        updated = np.zeros_like(centers)
        np.add.at(updated, a, x)
        updated /= np.maximum(counts[:, None], 1)
        empty = np.flatnonzero(counts == 0)
        if len(empty):
            distance = ((x - centers[a]) ** 2).sum(1)
            updated[empty] = x[np.argsort(distance)[-len(empty):]]
        centers = updated
    return centers


def pca_view(x, split, seed=50):
    """Fit a 2D PCA view to a label-blind reference subsample only."""
    from scipy.linalg import eigh
    rng = np.random.default_rng(seed)
    ref = np.flatnonzero(split == 0)
    ref = rng.choice(ref, min(len(ref), 2048), replace=False)
    sample = x[ref].astype(np.float64)
    mean = sample.mean(0)
    sample -= mean
    cov = sample.T @ sample
    values, vectors = eigh(cov, subset_by_index=[x.shape[1] - 2, x.shape[1] - 1])
    axes = vectors[:, ::-1]
    coords = (x - mean) @ axes
    fraction = float(np.maximum(values, 0).sum() / max(float(np.trace(cov)), 1e-12))
    return dict(points=np.round(coords, 5).tolist(), variance=fraction, reference_points=len(ref))


def textures():
    """Embed the installed Craftax textures; no JAX or texture-cache writes."""
    from PIL import Image
    import importlib.util
    package = importlib.util.find_spec('craftax')
    assets = Path(next(iter(package.submodule_search_locations))) / 'craftax_classic/assets'
    blocks = ['debug_tile', 'debug_tile', 'grass', 'water', 'stone', 'tree', 'wood',
              'path', 'coal', 'iron', 'diamond', 'table', 'furnace', 'sand', 'lava',
              'plant_on_grass', 'ripe_plant_on_grass']
    players = ['player-left', 'player-right', 'player-up', 'player-down', 'player-sleep']
    mobs = ['zombie', 'cow', 'skeleton', 'arrow-up']
    result = {}
    for name in set(blocks + players + mobs):
        img = Image.open(assets / (name + '.png')).convert('RGBA')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        result[name] = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
    return dict(images=result, blocks=blocks, players=players, mobs=mobs)


def build(source=DEFAULT_SOURCE, out=DEFAULT_OUT, sample_size=5000):
    source, out = Path(source).resolve(), Path(out).resolve()
    summary = json.loads((source / 'summary.json').read_text())
    settings = json.loads((source / 'settings.json').read_text())
    meta = json.loads((ROOT / 'concept_mapping/runs/master_symbolic_10b_5000episodes/dataset_metadata.json').read_text())
    with np.load(source / 'transition_bank.npz', allow_pickle=False) as file:
        bank = dict(file)
    validate_bank(bank)
    out.mkdir(parents=True, exist_ok=True)
    model_dir = out / 'models'
    model_dir.mkdir(exist_ok=True)
    mask = eligible(bank['rewards'], 'achievement_reward')
    tr = np.flatnonzero(mask & (bank['split_id'] == 0))
    te = np.flatnonzero(mask & (bank['split_id'] == 1))
    rng = np.random.default_rng(50)
    candidates = np.flatnonzero(mask)
    chosen = set(rng.choice(candidates, min(sample_size, len(candidates)), replace=False).tolist())
    runs = []
    # Refit centroids on +1 reference transitions. Filtering the old positive-
    # reward assignments would leave health recovery influencing the geometry.
    with np.load(source / 'features.npz', allow_pickle=False) as features:
        for method in METHODS:
            x = normalize(features[method][np.concatenate([tr, te])])
            for k in (32, 64, 128):
                for seed in ((50, 51, 52) if method == 'critic_delta' else (50,)):
                    row = dict(mode='achievement_reward', method=method, k=k, cluster_seed=seed)
                    key = f'achievement_reward_{method}_k{k}_seed{seed}_cpu25'
                    path = model_dir / (key + '.npz')
                    if path.exists():
                        with np.load(path, allow_pickle=False) as file:
                            cached = dict(file)
                        if not np.array_equal(cached['reference_indices'], tr) or not np.array_equal(cached['test_indices'], te):
                            raise ValueError('Saved +1 model split differs from source; use a fresh --out directory')
                    else:
                        print('Fitting', key, f'({len(tr):,} reference / {len(te):,} test)', flush=True)
                        centers = fit_kmeans(x[:len(tr)], k, seed, iterations=25)
                        cached = dict(centers=centers, reference_indices=tr, test_indices=te,
                                      reference_assignments=assignment(x[:len(tr)], centers),
                                      test_assignments=assignment(x[len(tr):], centers))
                        np.savez_compressed(path, **cached)
                    for split in ('reference', 'test'):
                        for cluster in range(k):
                            members = cached[split + '_indices'][cached[split + '_assignments'] == cluster]
                            chosen.update(rng.choice(members, min(2, len(members)), replace=False).tolist())
                    runs.append((row, key, cached))
    indices = np.array(sorted(chosen), np.int32)
    coords = {}; configs = []
    with np.load(source / 'features.npz', allow_pickle=False) as features:
        for method in METHODS:
            print('Preparing', method, flush=True)
            x = normalize(features[method][indices])
            coords[method] = pca_view(x, bank['split_id'][indices])
            for row, key, cache in runs:
                if row['method'] != method:
                    continue
                tr, te = cache['reference_indices'], cache['test_indices']
                a, b = cache['reference_assignments'], cache['test_assignments']
                all_assign = np.full(len(bank['rewards']), -1, np.int16)
                all_assign[tr], all_assign[te] = a, b
                display = all_assign[indices]
                valid = display >= 0
                distance = np.full(len(indices), -1., np.float32)
                distance[valid] = ((x[valid] - cache['centers'][display[valid]]) ** 2).sum(1)
                config = dict(id=key, mode=row['mode'], method=method, k=row['k'], seed=row['cluster_seed'], iterations=25,
                              reference_n=len(tr), test_n=len(te), fit_n=len(tr), cached=True,
                              assignments=display.tolist(), distances=np.round(distance, 6).tolist(),
                              **diagnostics(bank, tr, te, a, b, row['k']))
                configs.append(config)
    payload = dict(meta=dict(checkpoint=settings['checkpoint'], source=str(source),
                            episodes=summary['episodes'], reference_episodes=summary['reference_episodes'],
                            test_episodes=summary['test_episodes'], original_transitions=summary['original_transitions'],
                            selected_transitions=len(bank['rewards']), achievement_reward=int(mask.sum()),
                            achievement_reference=len(tr), achievement_test=len(te),
                            achievement_with_health_change=int(((bank['labels'] < 22) & ~mask).sum()),
                            inspection_sample=len(indices), sample_seed=50,
                            sampling='Random +1-reward transitions plus random members of each +1-reward cluster; no achievement balancing.',
                            policy_run=meta['run_path'], environment='Craftax-Classic-Symbolic-v1'),
                   names=[n.lower() for n in meta['achievement_names']] + EXTRA_NAMES,
                   stat_names=meta['inventory_names'] + meta['intrinsic_names'],
                   block_names=meta['block_type_names'],
                   action_names=['noop', 'left', 'right', 'up', 'down', 'do', 'sleep', 'place_stone',
                                 'place_table', 'place_furnace', 'place_plant', 'make_wood_pickaxe',
                                 'make_stone_pickaxe', 'make_iron_pickaxe', 'make_wood_sword',
                                 'make_stone_sword', 'make_iron_sword'],
                   rows=describe_rows(bank, indices), projections=coords, configs=configs, textures=textures())
    out.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
    html = (HERE / 'index.html').read_text().replace('/*__APP__*/', (HERE / 'app.js').read_text())
    html = html.replace('/*__DATA__*/', raw)
    (out / 'index.html').write_text(html)
    # Small sidecar links the optional local fitting server to exact display rows.
    (out / 'manifest.json').write_text(json.dumps(dict(source=str(source), indices=indices.tolist()), indent=2))
    print(f"Built {out / 'index.html'} ({len(html) / 1e6:.1f} MB), {len(indices):,} examples, {len(configs)} experiments", flush=True)
    return out / 'index.html'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--sample-size', type=int, default=5000)
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error('--sample-size must be positive')
    build(args.source, args.out, args.sample_size)

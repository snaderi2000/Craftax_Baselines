"""Label-free cluster fitting of frozen transition representations.

Sampling uses rewards and uniform random timesteps, NEVER achievement flags.
Flags label held-out diagnostics only. This screens representation geometry;
it does not train AD, discover prerequisites or improve the PPO policy.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
from concept_mapping.test_achievement_representations_offline import embed_bank, normalize, DATASETS

METHODS = ('critic_delta', 'actor_delta', 'raw_observation_delta',
           'random_actor_delta', 'action_inventory_delta')
EXTRA_NAMES = ['health_gain_only', 'health_loss_only', 'other_no_new_achievement',
               'multiple_new_achievements']


def sample_episode(observations, achievements, actions, rewards,
                   terminal_observation, terminal_achievements, uniform_n, seed):
    """Select without flags, then independently annotate actual transitions."""
    obs = np.asarray(observations, np.float32)
    n = len(obs)
    rewards = np.asarray(rewards, np.float32)
    if n == 0 or len(rewards) != n or len(actions) != n:
        raise ValueError('Malformed episode')
    rng = np.random.default_rng(seed)
    uniform = np.zeros(n, bool)
    uniform[rng.choice(n, min(n, uniform_n), replace=False)] = True
    # Positive AND negative health events are retained, as well as achievements.
    steps = np.flatnonzero(uniform | (np.abs(rewards) > 1e-7))
    after = np.empty((len(steps), obs.shape[1]), np.float32)
    continuing = steps + 1 < n
    after[continuing] = obs[steps[continuing] + 1]
    after[~continuing] = np.asarray(terminal_observation, np.float32)
    flags = np.asarray(achievements, bool)
    next_flags = np.concatenate([flags[1:], np.asarray(terminal_achievements, bool)[None]])
    if flags.shape != (n, 22) or np.any(flags & ~next_flags):
        raise ValueError('Malformed/nonmonotonic achievement flags')
    unlocked = (next_flags & ~flags)[steps]
    sizes = unlocked.sum(1)
    labels = np.full(len(steps), 24, np.int16)
    health_delta = after[:, 1335] - obs[steps, 1335]
    labels[health_delta > 1e-6] = 22
    labels[health_delta < -1e-6] = 23
    single = sizes == 1
    labels[single] = unlocked[single].argmax(1)
    labels[sizes > 1] = 25
    return dict(before=obs[steps], after=after, labels=labels, steps=steps,
                actions=np.asarray(actions, np.int32)[steps], rewards=rewards[steps],
                uniform_sample=uniform[steps], episode_length=n)


def load_selected_episode(job):
    row, uniform_n, seed = job
    dataset = int(row['dataset'])
    episode = int(row['episode'])
    with np.load(row['source_npz'], allow_pickle=False) as data:
        result = sample_episode(data['observations'], data['achievements'], data['actions'],
                                data['rewards'], data['terminal_observation'],
                                data['terminal_achievements'], uniform_n,
                                np.random.SeedSequence([seed, dataset, episode]))
    n = len(result['steps'])
    result['episode_uid'] = np.full(n, dataset * 100000 + episode, np.int32)
    result['split_id'] = np.full(n, row['split'] == 'test', np.int8)
    return result


def contingency(labels, assignments, k, classes=26):
    return np.bincount(np.asarray(labels, int) * k + np.asarray(assignments, int),
                       minlength=classes * k).reshape(classes, k)


def partition_metrics(table):
    """ARI, NMI, homogeneity and completeness, without sklearn dependency."""
    table = np.asarray(table, np.float64)
    n = table.sum()
    row, col = table.sum(1), table.sum(0)
    if n <= 1:
        return dict(ari=1., nmi=1., homogeneity=1., completeness=1.)
    p = table / n
    expected = row[:, None] * col[None, :] / n**2
    nonzero = p > 0
    mi = float((p[nonzero] * np.log(p[nonzero] / expected[nonzero])).sum())
    def entropy(counts):
        probabilities = counts[counts > 0] / n
        return float(-(probabilities * np.log(probabilities)).sum())
    hr, hc = entropy(row), entropy(col)
    choose2 = lambda a: (a * (a - 1) / 2).sum()
    index, rows, cols = choose2(table), choose2(row), choose2(col)
    chance = rows * cols / (n * (n - 1) / 2)
    denominator = .5 * (rows + cols) - chance
    return dict(ari=float((index - chance) / denominator) if abs(denominator) > 1e-12 else 1.,
                nmi=2 * mi / (hr + hc) if hr + hc > 0 else 1.,
                homogeneity=mi / hr if hr > 0 else 1., completeness=mi / hc if hc > 0 else 1.)


def classify_clusters(train_labels, train_assignments, test_labels, test_assignments, k):
    """Post-fit semantic majority labels ONLY; no labels affect centroids."""
    reference_table = contingency(train_labels, train_assignments, k)
    mapping = reference_table.argmax(0)
    mapping[reference_table.sum(0) == 0] = 24
    predicted = mapping[test_assignments]
    confusion = np.bincount(np.asarray(test_labels, int) * 26 + predicted,
                            minlength=26**2).reshape(26, 26)
    tp = confusion.diagonal().astype(float)
    support = confusion.sum(1)
    predicted_n = confusion.sum(0)
    precision = np.divide(tp, predicted_n, out=np.zeros(26), where=predicted_n > 0)
    recall = np.divide(tp, support, out=np.zeros(26), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted_n, out=np.zeros(26), where=(support + predicted_n) > 0)
    table = contingency(test_labels, test_assignments, k)
    mask = np.asarray(test_labels) < 22
    metrics = partition_metrics(table)
    metrics.update(mapped_accuracy=float((predicted == test_labels).mean()),
                   achievement_accuracy=float((predicted[mask] == test_labels[mask]).mean()) if mask.any() else None,
                   achievement_macro_f1=float(f1[:22][support[:22] > 0].mean()),
                   achievement_macro_recall=float(recall[:22][support[:22] > 0].mean()),
                   heldout_cluster_purity=float(table.max(0).sum() / table.sum()),
                   occupied_reference_clusters=int((reference_table.sum(0) > 0).sum()),
                   occupied_test_clusters=int((table.sum(0) > 0).sum()))
    return metrics, precision, recall, f1, support, mapping, reference_table, table, predicted


def fit_kmeans_unlabeled(x, k, seed, iterations=25):
    """GPU Lloyd k-means; k-means++ initialization uses an unlabeled subsample."""
    import jax
    import jax.numpy as jnp
    x = jnp.asarray(normalize(x))
    rng = np.random.default_rng(seed)
    init_indices = rng.choice(len(x), min(len(x), 8192), replace=False)
    sample = x[init_indices]
    @jax.jit
    def fit(x, sample):
        keys = jax.random.split(jax.random.PRNGKey(seed), k + 1)
        first = jax.random.randint(keys[0], (), 0, len(sample))
        centers = jnp.zeros((k, x.shape[1]), jnp.float32).at[0].set(sample[first])
        distances = ((sample - sample[first])**2).sum(1)
        def initialize(i, carry):
            centers, distances = carry
            index = jax.random.categorical(keys[i], jnp.log(jnp.maximum(distances, 1e-12)))
            point = sample[index]
            distances = jnp.minimum(distances, ((sample - point)**2).sum(1))
            return centers.at[i].set(point), distances
        centers, _ = jax.lax.fori_loop(1, k, initialize, (centers, distances))
        xnorm = (x**2).sum(1)[:, None]
        def update(_, centers):
            distances = xnorm + (centers**2).sum(1)[None] - 2 * jnp.matmul(
                x, centers.T, precision=jax.lax.Precision.HIGHEST)
            assignments = distances.argmin(1)
            counts = jnp.bincount(assignments, length=k)
            sums = jax.ops.segment_sum(x, assignments, num_segments=k)
            return jnp.where((counts > 0)[:, None], sums / jnp.maximum(1, counts)[:, None], centers)
        return jax.lax.fori_loop(0, iterations, update, centers)
    return np.asarray(fit(x, sample))


def assign_clusters(x, centers, batch_size=4096):
    import jax
    import jax.numpy as jnp
    centers = jnp.asarray(centers)
    @jax.jit
    def assign(batch):
        return ((batch**2).sum(1)[:, None] + (centers**2).sum(1)[None] -
                2 * jnp.matmul(batch, centers.T, precision=jax.lax.Precision.HIGHEST)).argmin(1)
    x = normalize(x)
    return np.concatenate([np.asarray(assign(jnp.asarray(x[i:i + batch_size])))
                           for i in range(0, len(x), batch_size)])


def save_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'})
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    names = [r['achievement'] for r in rows if r['mode'] == 'positive_reward'
             and r['method'] == 'critic_delta' and r['k'] == 64 and r['cluster_seed'] == 50]
    for ax, mode in zip(axes, ('positive_reward', 'mixed_reward_and_uniform')):
        for method, color in (('critic_delta', '#e41a1c'), ('actor_delta', '#2462f2'),
                              ('raw_observation_delta', '#777777'), ('action_inventory_delta', '#2ca02c')):
            selected = [r for r in rows if r['mode'] == mode and r['method'] == method
                        and r['k'] == 64 and r['cluster_seed'] == 50]
            ax.plot(range(22), [100*r['f1'] for r in selected[:22]], marker='o', markersize=3,
                    color=color, label=method.replace('_', ' '))
        ax.set_title(mode.replace('_', ' ') + ' (64 unlabeled clusters)')
        ax.set_ylabel('Held-out F1 (%)'); ax.set_ylim(-2, 102); ax.grid(alpha=.2)
        ax.legend(fontsize=8, ncol=2)
    axes[-1].set_xticks(range(22), names[:22], rotation=55, ha='right', fontsize=8)
    fig.tight_layout()
    for extension in ('pdf', 'png'):
        fig.savefig(out/f'achievement_cluster_f1.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'concept_mapping/runs/achievement_clustering_offline_10b_batch2000')
    parser.add_argument('--selection', type=Path, default=ROOT/'concept_mapping/runs/achievement_representation_offline_10b_batch2000/episode_selection.csv')
    parser.add_argument('--uniform_steps_per_episode', type=int, default=64)
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=25)
    parser.add_argument('--resume', action='store_true', help='Reuse this analysis cache; never overwrite datasets')
    args = parser.parse_args(); out = args.out
    if out.exists() and not args.resume:
        raise RuntimeError(f'Refusing overwrite {out}; use --resume only for this analysis output')
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    def progress(stage, **extra):
        (out/'progress.json').write_text(json.dumps(dict(stage=stage, elapsed_seconds=time.time()-started, **extra), indent=2)+'\n')
    metadata = json.loads((DATASETS[0]/'dataset_metadata.json').read_text())
    names = [n.lower() for n in metadata['achievement_names']] + EXTRA_NAMES
    settings = dict(selection=str(args.selection.resolve()), uniform_steps_per_episode=args.uniform_steps_per_episode,
                    analysis_seed=args.seed, iterations=args.iterations, checkpoint=metadata['checkpoint'],
                    methods=list(METHODS), clusters=[32, 64, 128], primary_k=64)
    settings_file = out/'settings.json'
    if args.resume and settings_file.exists() and json.loads(settings_file.read_text()) != settings:
        raise ValueError('Resume settings differ from cached analysis')
    settings_file.write_text(json.dumps(settings, indent=2)+'\n')
    if args.resume and (out/'transition_bank.npz').exists():
        bank = dict(np.load(out/'transition_bank.npz', allow_pickle=False))
    else:
        progress('extracting')
        with args.selection.open() as file:
            selection = list(csv.DictReader(file))
        save_csv(out/'episode_selection.csv', selection)
        jobs = [(row, args.uniform_steps_per_episode, args.seed) for row in selection]
        chunks = []; total_original = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, chunk in enumerate(pool.map(load_selected_episode, jobs), 1):
                total_original += chunk.pop('episode_length'); chunks.append(chunk)
                if i % 100 == 0:
                    print(f'Extracted {i}/{len(jobs)} episodes', flush=True)
        bank = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        bank['original_transition_count'] = np.asarray(total_original)
        np.savez_compressed(out/'transition_bank.npz', **bank)
        del chunks
    train = bank['split_id'] == 0; test = ~train
    if set(bank['episode_uid'][train]) & set(bank['episode_uid'][test]):
        raise AssertionError('Episode leakage')
    if args.resume and (out/'features.npz').exists():
        features = dict(np.load(out/'features.npz'))
        validation = json.loads((out/'encoder_validation.json').read_text())
    else:
        progress('embedding', transitions=len(train))
        all_features, validation = embed_bank(bank, metadata, 512, args.seed)
        features = {method: all_features[method] for method in METHODS}
        np.savez_compressed(out/'features.npz', **features)
        (out/'encoder_validation.json').write_text(json.dumps(validation, indent=2)+'\n')
        del all_features
    summary = []; class_rows = []; compositions = []
    for mode in ('positive_reward', 'mixed_reward_and_uniform'):
        eligible = bank['rewards'] > 1e-7 if mode == 'positive_reward' else np.ones(len(train), bool)
        tr = np.flatnonzero(train & eligible); te = np.flatnonzero(test & eligible)
        print(f'{mode}: reference={len(tr)}, heldout={len(te)}', flush=True)
        for method in METHODS:
            for k in (32, 64, 128):
                # Extra initializations screen the main learned critic for instability.
                for cluster_seed in ((50, 51, 52) if method == 'critic_delta' else (50,)):
                    key = dict(mode=mode, method=method, k=k, cluster_seed=cluster_seed)
                    progress('clustering', **key)
                    prefix = f'{mode}_{method}_k{k}_seed{cluster_seed}'
                    cached = out/f'{prefix}.npz'
                    if args.resume and cached.exists():
                        cached_data = np.load(cached)
                        centers = cached_data['centers']; ref_assign = cached_data['reference_assignments']
                        test_assign = cached_data['test_assignments']
                    else:
                        centers = fit_kmeans_unlabeled(features[method][tr], k, cluster_seed, args.iterations)
                        ref_assign = assign_clusters(features[method][tr], centers)
                        test_assign = assign_clusters(features[method][te], centers)
                    result = classify_clusters(bank['labels'][tr], ref_assign, bank['labels'][te], test_assign, k)
                    metrics, precision, recall, f1, support, mapping, ref_table, test_table, predicted = result
                    np.savez_compressed(cached, centers=centers, reference_indices=tr, test_indices=te,
                        reference_assignments=ref_assign, test_assignments=test_assign,
                        semantic_mapping=mapping, reference_contingency=ref_table, test_contingency=test_table)
                    row = dict(**key, reference_n=len(tr), test_n=len(te), **metrics)
                    # Uniform-only queries remove the reward-event enrichment from evaluation.
                    uniform = bank['uniform_sample'][te]
                    if uniform.any():
                        u = classify_clusters(bank['labels'][tr], ref_assign, bank['labels'][te][uniform], test_assign[uniform], k)[0]
                        row.update(uniform_test_n=int(uniform.sum()), uniform_achievement_macro_f1=u['achievement_macro_f1'])
                    summary.append(row)
                    for label, name in enumerate(names):
                        class_rows.append(dict(**key, achievement=name, reference_n=int(ref_table[label].sum()),
                            test_n=int(support[label]), precision=float(precision[label]), recall=float(recall[label]),
                            f1=float(f1[label]), mapped_clusters=int((mapping == label).sum())))
                    for cluster in range(k):
                        for label, name in enumerate(names):
                            if ref_table[label, cluster] or test_table[label, cluster]:
                                compositions.append(dict(**key, cluster=cluster, achievement=name,
                                    reference_n=int(ref_table[label, cluster]), test_n=int(test_table[label, cluster]),
                                    mapped_achievement=names[mapping[cluster]]))
                    print(prefix, {m: round(metrics[m], 4) for m in
                          ('achievement_macro_f1', 'achievement_macro_recall', 'nmi', 'ari')}, flush=True)
                    save_csv(out/'summary.csv', summary)
    save_csv(out/'per_achievement.csv', class_rows)
    save_csv(out/'cluster_composition.csv', compositions)
    payload = dict(**settings, episodes=int(len(np.unique(bank['episode_uid']))),
        reference_episodes=int(len(np.unique(bank['episode_uid'][train]))),
        test_episodes=int(len(np.unique(bank['episode_uid'][test]))),
        original_transitions=int(bank['original_transition_count']), selected_transitions=len(train),
        positive_reward_transitions=int((bank['rewards'] > 1e-7).sum()),
        validation=validation, elapsed_seconds=time.time()-started, results=summary,
        protocol='Episode-disjoint frozen-feature k-means; majority semantic mapping after fitting on reference only.',
        labels_used_for='Post-fit diagnostics and semantic mapping ONLY; never sampling, embedding, initialization, fitting or selecting k.',
        policy_training=False, encoder_training=False, new_rollouts=False,
        caveats=['Mature 10B policy, not current early-training representations.',
            'Positive-reward screen assumes a scalar reward event signal, not semantic achievement IDs.',
            'Mixed sampling retains every nonzero reward and up to 64 uniform steps per episode; it is enriched, not natural frequency.',
            'Uniform-only held-out diagnostics reduce enrichment but have fewer rare-achievement examples.',
            'First-achievement flags label diagnostics; repeated unrewarded executions can be visually identical.',
            'Reference majority mapping is labeled evaluation, not a mechanism supplied to the policy.',
            'Fixed 32/64/128 cluster sweep and critic initialization seeds; no k selection by held-out labels.',
            'Geometry alone does not establish causal knowledge or useful practice gains.'])
    (out/'summary.json').write_text(json.dumps(payload, indent=2)+'\n')
    plot_results(class_rows, out)
    progress('complete', selected_transitions=len(train))
    print('Complete:', out, flush=True)


if __name__ == '__main__':
    main()

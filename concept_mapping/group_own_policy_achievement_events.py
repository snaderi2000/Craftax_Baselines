"""Own-policy reward-event collection and label-free POT/prototype grouping.

No lead-ups, full trajectories, practice resets or training. Semantic labels
are stored separately and are read only AFTER fitting numerical groups.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
from concept_mapping.compare_transport_achievement_groups import (
    achievement_reward_proxy, build_transport_groups, nearest_prototypes,
    audit_classifier, bootstrap_metrics, save_csv, save_inference_model)
from concept_mapping.test_achievement_representations_offline import embed_bank


def episode_quotas(episodes, workers):
    if episodes < 2 or workers < 1:
        raise ValueError('Need at least two episodes and one worker')
    quotas = np.full(workers, episodes // workers, np.int32)
    quotas[:episodes % workers] += 1
    return quotas, np.concatenate(([0], np.cumsum(quotas)[:-1])).astype(np.int32)


def split_episodes(episodes, seed):
    split = np.ones(episodes, np.int8)
    split[np.random.default_rng(seed).permutation(episodes)[:int(.7 * episodes)]] = 0
    return split


def collect(args, out):
    import jax
    import jax.numpy as jnp
    from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer
    from craftax.craftax_env import make_craftax_env_from_name
    from concept_mapping.render_ppo_episodes import _load_wandb_config, _init_network

    if (out / 'event_bank.npz').exists():
        raise FileExistsError('Event bank already exists; use --stage group to reuse it')
    start = time.monotonic()
    run = Path(args.run).resolve()
    config = _load_wandb_config(str(run))
    env = make_craftax_env_from_name(config['ENV_NAME'], auto_reset=False)
    params = env.default_params.replace(max_timesteps=args.max_steps)
    net = _init_network(config, env, params)
    manager = CheckpointManager(str(run / 'files/policies'), PyTreeCheckpointer(),
                               CheckpointManagerOptions(create=False))
    weights = manager.restore(args.checkpoint)['params']
    n = args.workers
    quotas, offsets = episode_quotas(args.episodes, n)
    split = split_episodes(args.episodes, args.split_seed)
    reset = jax.vmap(env.reset, in_axes=(0, None))
    step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    key, rk = jax.random.split(jax.random.PRNGKey(args.seed))
    obs, state = reset(jax.random.split(rk, n), params)

    def transition(carry, unused):
        key, obs, state = carry
        key, ak, ek, rk = jax.random.split(key, 4)
        pi, _ = net.apply(weights, obs)
        actions = pi.sample(seed=ak)
        after, next_state, reward, done, _ = step(jax.random.split(ek, n), state, actions, params)
        # Output the TRUE successor, including terminal transitions, before reset.
        # Flags are evaluation sidecars, never used by the event selector.
        new_flags = next_state.achievements & ~state.achievements
        records = (obs, after, actions, reward, done, new_flags)

        def reset_finished(unused):
            robs, rstate = reset(jax.random.split(rk, n), params)
            def choose(fresh, old):
                mask = done.reshape((n,) + (1,) * (old.ndim - 1))
                return jnp.where(mask, fresh, old)
            return jnp.where(done[:, None], robs, after), jax.tree_util.tree_map(choose, rstate, next_state)
        continuing_obs, continuing_state = jax.lax.cond(
            done.any(), reset_finished, lambda _: (after, next_state), operand=None)
        return (key, continuing_obs, continuing_state), records

    block = jax.jit(lambda carry: jax.lax.scan(transition, carry, None, length=args.block_steps))
    completed = np.zeros(n, np.int32)
    lengths = np.zeros(n, np.int32)
    returns = np.zeros(n, np.float64)
    chunks = []
    label_chunks = []
    episodes = []
    event_count = transitions = terminal_events = compound_events = 0
    last_log = start
    carry = (key, obs, state)
    print('Collecting ordinary stochastic-policy episodes; first block includes JIT compilation', flush=True)
    while len(episodes) < args.episodes:
        carry, records = block(carry)
        before, after, actions, rewards, done, flags = jax.tree_util.tree_map(np.asarray, records)
        shape = rewards.shape
        proxy = achievement_reward_proxy(before.reshape(-1, 1345), after.reshape(-1, 1345),
                                         rewards.reshape(-1)).reshape(shape)
        candidate = proxy > .5
        # Audit detection against privileged flags without using them to select.
        counts = flags.sum(-1)
        if not np.allclose(proxy, counts, atol=1e-5):
            raise AssertionError('Reward decomposition does not match new achievement count')
        for t in range(args.block_steps):
            active = completed < quotas
            workers = np.flatnonzero(active & candidate[t])
            transitions += int(active.sum())
            if len(workers):
                uids = offsets[workers] + completed[workers]
                number = len(workers)
                chunks.append(dict(before=before[t, workers], after=after[t, workers],
                    actions=actions[t, workers], rewards=rewards[t, workers],
                    episode_uid=uids, steps=lengths[workers].copy(),
                    terminal=done[t, workers], split_id=split[uids],
                    event_id=np.arange(event_count, event_count + number, dtype=np.int32)))
                labels = np.where(counts[t, workers] == 1, flags[t, workers].argmax(-1), 25)
                label_chunks.append(dict(event_id=chunks[-1]['event_id'], labels=labels.astype(np.int16),
                                         new_achievement_flags=flags[t, workers]))
                event_count += number
                terminal_events += int(done[t, workers].sum())
                compound_events += int((counts[t, workers] > 1).sum())
            lengths[active] += 1
            returns[active] += rewards[t, active]
            for worker in np.flatnonzero(active & done[t]):
                uid = int(offsets[worker] + completed[worker])
                episodes.append(dict(episode_uid=uid, worker=int(worker),
                    worker_episode=int(completed[worker]), split='reference' if split[uid] == 0 else 'test',
                    length=int(lengths[worker]), return_=float(returns[worker]),
                    hit_cap=bool(lengths[worker] >= args.max_steps)))
                completed[worker] += 1
                lengths[worker] = 0
                returns[worker] = 0
        if time.monotonic() - last_log > 30 or len(episodes) == args.episodes:
            print(f'{len(episodes)}/{args.episodes} episodes; {transitions:,} transitions; '
                  f'{event_count:,} achievement events; elapsed {time.monotonic()-start:.1f}s', flush=True)
            last_log = time.monotonic()
    if not chunks:
        raise RuntimeError('No achievement events collected')
    bank = {name: np.concatenate([c[name] for c in chunks]) for name in chunks[0]}
    audit = {name: np.concatenate([c[name] for c in label_chunks]) for name in label_chunks[0]}
    if len(np.unique(bank['event_id'])) != event_count:
        raise AssertionError('Duplicate event IDs')
    if set(bank['episode_uid'][bank['split_id'] == 0]) & set(bank['episode_uid'][bank['split_id'] == 1]):
        raise AssertionError('Episode split leakage')
    np.savez_compressed(out / 'event_bank.npz', **bank)
    np.savez_compressed(out / 'evaluation_labels.npz', **audit)
    save_csv(out / 'episode_manifest.csv', sorted(episodes, key=lambda r: r['episode_uid']))
    metadata = dict(run_path=str(run), checkpoint=args.checkpoint, training_seed=config.get('SEED'),
        collection_seed=args.seed, split_seed=args.split_seed, environment=config['ENV_NAME'],
        reset='standard normal environment; no practice/optimistic reset', stochastic_policy=True,
        workers=n, block_steps=args.block_steps, episodes=args.episodes, transitions=transitions,
        max_episode_steps=args.max_steps, capped_episodes=sum(e['hit_cap'] for e in episodes),
        median_episode_length=float(np.median([e['length'] for e in episodes])),
        achievement_events=event_count, terminal_events=terminal_events, compound_events=compound_events,
        reference_episodes=int((split == 0).sum()), test_episodes=int((split == 1).sum()),
        event_filter='reward minus 0.1*health change > 0.5; no labels used',
        fields_saved=list(bank), labels_separate=True, leadups_saved=False, full_trajectories_saved=False,
        true_terminal_successors=True, wall_seconds=time.monotonic()-start)
    (out / 'collection_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2), flush=True)


def group(args, out):
    from craftax.craftax_classic.constants import Achievement
    names = [a.name.lower() for a in Achievement]
    metadata = json.loads((out / 'collection_metadata.json').read_text())
    bank = dict(np.load(out / 'event_bank.npz', allow_pickle=False))
    if 'labels' in bank or 'new_achievement_flags' in bank:
        raise AssertionError('Labels must not be stored in the fitting bank')
    feature_path = out / 'features.npz'
    if feature_path.exists():
        features = dict(np.load(feature_path, allow_pickle=False))
    else:
        features, validation = embed_bank(bank, metadata, 512, args.seed)
        features = {k: features[k] for k in ('actor_delta', 'critic_delta')}
        np.savez_compressed(feature_path, **features)
        (out / 'encoder_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    reference = np.flatnonzero(bank['split_id'] == 0)
    test = np.flatnonzero(bank['split_id'] == 1)
    fits = []
    # Complete every unlabeled fit BEFORE loading the evaluation-label file.
    for representation in ('critic_delta', 'actor_delta'):
        for seed in args.group_seeds:
            folder = out / 'groups' / f'{representation}_seed{seed}'
            folder.mkdir(parents=True, exist_ok=True)
            x = features[representation]
            centers, assignment, confidence, anchor, uid, residual, audit = build_transport_groups(
                x[reference], bank['episode_uid'][reference], seed, args.transport_workers)
            test_groups = nearest_prototypes(x[test], centers)
            reference_decoded = nearest_prototypes(x[reference], centers)
            save_inference_model(folder / 'model.npz', centers, representation, metadata, 'transport_anchor')
            np.savez_compressed(folder / 'group_membership.npz', reference_event_id=bank['event_id'][reference],
                reference_group=assignment, reference_decoded_group=reference_decoded,
                test_event_id=bank['event_id'][test], test_group=test_groups, confidence=confidence,
                anchor_event_id=bank['event_id'][reference[anchor]], **audit)
            fits.append((representation, seed, folder, assignment, reference_decoded,
                         test_groups, anchor, uid, residual, len(centers), audit))
            print(f'Fitted {representation} seed {seed}: {len(centers)} unlabeled groups, anchor episode {uid}', flush=True)
    evaluation = dict(np.load(out / 'evaluation_labels.npz', allow_pickle=False))
    np.testing.assert_array_equal(evaluation['event_id'], bank['event_id'])
    labels = evaluation['labels']
    frequencies = evaluation['new_achievement_flags'].sum(0)
    label_rows = [dict(achievement=name, total_events=int(frequencies[i]),
        reference_events=int((labels[reference] == i).sum()), test_events=int((labels[test] == i).sum()),
        episode_rate=float(frequencies[i] / metadata['episodes'])) for i, name in enumerate(names)]
    save_csv(out / 'evaluation_achievement_counts.csv', label_rows)
    rows = []
    for representation, seed, folder, assignment, decoded, test_groups, anchor, uid, residual, k, audit in fits:
        result, precision, recall, f1, support, mapping, predicted, table = audit_classifier(
            labels[reference], assignment, labels[test], test_groups, k)
        ci, _, _ = bootstrap_metrics(labels[test], predicted, bank['episode_uid'][test],
                                    args.split_seed + seed, args.bootstrap_samples)
        result.update(ci)
        result.update(representation=representation, grouping_seed=seed, groups=k, anchor_episode=uid,
            reference_events=len(reference), test_events=len(test),
            reference_observed_achievement_types=len(np.unique(labels[reference][labels[reference] < 22])),
            test_observed_achievement_types=int((support[:22] > 0).sum()),
            macro_f1_definition='equal weight over achievement classes observed in test, not necessarily all 22',
            macro_f1_all22=float(f1[:22].mean()), max_transport_capacity_residual=residual,
            transport_conflict_episodes=int(audit['conflict_episodes']),
            anchor_semantic_labels_for_audit_only=labels[reference[anchor]].tolist(),
            bootstrap_samples=args.bootstrap_samples)
        (folder / 'evaluation_summary.json').write_text(json.dumps(result, indent=2) + '\n')
        save_csv(folder / 'evaluation_per_achievement.csv', [dict(achievement=name, support=int(support[i]),
            reference_support=int((labels[reference] == i).sum()), precision=float(precision[i]),
            recall=float(recall[i]), f1=float(f1[i]), mapped_groups=int((mapping == i).sum()))
            for i, name in enumerate(names)])
        # Frequency estimates use all reference events decoded individually,
        # as they would be during online assignment, not just accepted POT edges.
        rarity = []
        for group_id in range(k):
            inds = reference[decoded == group_id]
            semantic = int(mapping[group_id])
            rarity.append(dict(group_id=group_id, reference_events=len(inds),
                reference_episodes=int(len(np.unique(bank['episode_uid'][inds]))),
                reference_episode_rate=float(len(np.unique(bank['episode_uid'][inds])) / metadata['reference_episodes']),
                audit_majority_achievement=names[semantic] if semantic < 22 else 'unknown_or_compound',
                audit_transport_purity=float(table[semantic, group_id] / max(1, table[:, group_id].sum()))))
        save_csv(folder / 'evaluation_group_rarity.csv', sorted(rarity, key=lambda r: r['reference_episode_rate']))
        row = {key: value for key, value in result.items() if not isinstance(value, (list, dict))}
        rows.append(row)
        print(f'{representation} seed{seed}: accuracy {result["accuracy"]:.2%}; '
              f'macro F1 {result["macro_f1"]:.2%}; mapped achievements '
              f'{result["semantic_achievements_represented"]}/{result["reference_observed_achievement_types"]}', flush=True)
    save_csv(out / 'grouping_summary.csv', rows)
    summary = dict(collection=metadata, fitting='frozen own-policy hidden after-minus-before features; '
        'POT to a maximal-event reference episode; group prototypes with nearest-prototype held-out decoder',
        no_label_supervision=True, reference_events=len(reference), test_events=len(test),
        observed_achievements=[names[i] for i in range(22) if frequencies[i] > 0],
        unobserved_achievements=[names[i] for i in range(22) if frequencies[i] == 0],
        evaluation_achievement_counts=label_rows, results=[json.loads((fit[2] / 'evaluation_summary.json').read_text()) for fit in fits])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('Saved own-policy grouping results:', out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default=str(ROOT / 'wandb/run-20261013_011547-gtzkxoth'))
    parser.add_argument('--checkpoint', type=int, default=49995776)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('collect', 'group', 'both'), default='both')
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--block-steps', type=int, default=32)
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=3050)
    parser.add_argument('--split-seed', type=int, default=4050)
    parser.add_argument('--group-seeds', type=int, nargs='+', default=[50, 51, 52])
    parser.add_argument('--transport-workers', type=int, default=4)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage in ('collect', 'both'):
        collect(args, args.out)
    if args.stage in ('group', 'both'):
        group(args, args.out)


if __name__ == '__main__':
    main()

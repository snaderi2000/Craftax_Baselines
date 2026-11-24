"""Frozen-feature AD-style matching vs k-means; no auxiliary/PPO training.

POT matching is the published AD rule, not the full AD algorithm. A maximum-
reward-count reference episode supplies persistent group IDs (our extension).
Ground-truth achievement labels only audit groups after fitting. Models saved
for inference contain numerical group IDs, not semantic achievement labels.
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
from concept_mapping.test_achievement_representations_offline import normalize, embed_bank, DATASETS
from concept_mapping.test_achievement_clustering_offline import fit_kmeans_unlabeled

UPSTREAM = 'https://github.com/snu-mllab/Achievement-Distillation/blob/main/achievement_distillation/algorithm/ppo_ad.py'
GROUPINGS = ('transport_anchor', 'kmeans22', 'kmeans32', 'kmeans64', 'kmeans128')


def achievement_reward_proxy(before, after, rewards):
    """Known Classic health reward is .1 per point; no achievement flags used."""
    delta_health = np.rint(np.asarray(after)[:, 1335] * 10) - np.rint(np.asarray(before)[:, 1335] * 10)
    return np.asarray(rewards) - .1 * delta_health


def get_transport_solver():
    try:
        from ot.partial import entropic_partial_wasserstein
    except ImportError:
        # Reuse an existing dependency without installing/changing either env.
        existing = Path('/home/shawheen/miniconda3/envs/ad-crafter/lib/python3.10/site-packages')
        if not existing.is_dir():
            raise RuntimeError('Install Python Optimal Transport (POT), or supply its environment')
        sys.path.append(str(existing))
        from ot.partial import entropic_partial_wasserstein
    return entropic_partial_wasserstein


def match_events(source, target, solver=None):
    """Exact upstream AD hyperparameters: unit masses, reg=.05, 100 iterations, T>.5."""
    if solver is None:
        solver = get_transport_solver()
    x, y = normalize(source).astype(np.float64), normalize(target).astype(np.float64)
    cost = np.maximum(0, 1 - x @ y.T)
    plan = solver(np.ones(len(x)), np.ones(len(y)), cost, reg=.05, numItermax=100)
    if not np.isfinite(plan).all():
        raise ValueError('Nonfinite transport plan')
    accepted = plan > .5
    raw_source, raw_target = np.nonzero(accepted)
    conflicts = bool((accepted.sum(0) > 1).any() or (accepted.sum(1) > 1).any())
    # Preserve raw AD edges for auditing. Persistent single-group membership
    # additionally requires conflict resolution; this is our decoder extension.
    groups = np.full(len(x), -1, np.int32)
    occupied = np.zeros(len(y), bool)
    order = np.argsort(-plan[raw_source, raw_target], kind='stable')
    for index in order:
        i, j = raw_source[index], raw_target[index]
        if groups[i] < 0 and not occupied[j]:
            groups[i] = j; occupied[j] = True
    confidence = plan.max(1)
    residual = max(0., float(plan.sum(1).max() - 1), float(plan.sum(0).max() - 1))
    return groups, confidence, residual, dict(raw_source=raw_source, raw_target=raw_target,
        raw_confidence=plan[raw_source, raw_target], conflict=conflicts)


def nearest_prototypes(features, centers, batch_size=4096):
    """Shared Euclidean decoder on normalized event vectors for both methods."""
    x = normalize(features)
    cnorm = (centers**2).sum(1)[None]
    assignments = []
    for start in range(0, len(x), batch_size):
        block = x[start:start + batch_size]
        distances = (block**2).sum(1)[:, None] + cnorm - 2 * block @ centers.T
        assignments.append(distances.argmin(1))
    return np.concatenate(assignments).astype(np.int32)


def select_anchor(episode_uids, seed):
    uids, counts = np.unique(episode_uids, return_counts=True)
    # Uses only observed reward-event counts, not the number/names of achievements.
    selected = np.random.default_rng(seed).choice(uids[counts == counts.max()])
    return np.flatnonzero(episode_uids == selected), int(selected)


def match_episodes(features, episode_uids, anchor_features, workers, own_anchor_uid=None):
    solver = get_transport_solver()
    episodes = [np.flatnonzero(episode_uids == uid) for uid in np.unique(episode_uids)]
    def work(indices):
        if own_anchor_uid is not None and episode_uids[indices[0]] == own_anchor_uid:
            identity = np.arange(len(indices), dtype=np.int32)
            return indices, identity, np.ones(len(indices)), 0., dict(raw_source=identity,
                raw_target=identity, raw_confidence=np.ones(len(indices)), conflict=False)
        groups, confidence, residual, audit = match_events(features[indices], anchor_features, solver)
        return indices, groups, confidence, residual, audit
    assignments = np.full(len(features), -1, np.int32)
    confidences = np.zeros(len(features))
    residual = 0.; raw_source = []; raw_target = []; raw_confidence = []; conflicts = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for indices, groups, confidence, error, audit in pool.map(work, episodes):
            assignments[indices] = groups; confidences[indices] = confidence
            residual = max(residual, error)
            raw_source.extend(indices[audit['raw_source']].tolist())
            raw_target.extend(audit['raw_target'].tolist())
            raw_confidence.extend(audit['raw_confidence'].tolist())
            conflicts += int(audit['conflict'])
    return assignments, confidences, residual, dict(raw_source=np.asarray(raw_source, np.int32),
        raw_target=np.asarray(raw_target, np.int32), raw_confidence=np.asarray(raw_confidence),
        conflict_episodes=np.asarray(conflicts), episodes=np.asarray(len(episodes)))


def build_transport_groups(features, episode_uids, seed, workers):
    x = normalize(features)
    anchor, uid = select_anchor(episode_uids, seed)
    assignment, confidence, residual, audit = match_episodes(x, episode_uids, x[anchor], workers, uid)
    centers = np.stack([x[assignment == i].mean(0) for i in range(len(anchor))])
    return centers, assignment, confidence, anchor, uid, residual, audit


def audit_classifier(reference_labels, reference_groups, test_labels, test_groups, k):
    """Reference majority naming is for evaluation only, after unsupervised fitting."""
    valid = reference_groups >= 0
    table = np.bincount(reference_labels[valid].astype(int) * k + reference_groups[valid],
                       minlength=26 * k).reshape(26, k)
    mapping = table.argmax(0)
    mapping[table.sum(0) == 0] = 26
    predicted = np.full(len(test_groups), 26, np.int32)  # abstention/unknown
    known = test_groups >= 0
    predicted[known] = mapping[test_groups[known]]
    confusion = np.bincount(test_labels.astype(int) * 27 + predicted,
                            minlength=27**2).reshape(27, 27)
    support = confusion.sum(1); predicted_n = confusion.sum(0)
    tp = confusion.diagonal().astype(float)
    precision = np.divide(tp, predicted_n, out=np.zeros(27), where=predicted_n > 0)
    recall = np.divide(tp, support, out=np.zeros(27), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted_n, out=np.zeros(27), where=support + predicted_n > 0)
    metrics = dict(accuracy=float((predicted == test_labels).mean()),
        macro_f1=float(f1[:22][support[:22] > 0].mean()), macro_recall=float(recall[:22][support[:22] > 0].mean()),
        assignment_coverage=float(known.mean()), reference_assignment_coverage=float(valid.mean()),
        semantic_achievements_represented=int(len(set(mapping[(mapping >= 0) & (mapping < 22)].tolist()))))
    return metrics, precision, recall, f1, support, mapping, predicted, table


def bootstrap_metrics(labels, predicted, episode_uids, seed=5050, samples=2000):
    episodes, inv = np.unique(episode_uids, return_inverse=True)
    n = len(episodes)
    truth = np.zeros((n, 27)); guesses = np.zeros((n, 27)); correct = np.zeros((n, 27))
    np.add.at(truth, (inv, labels), 1)
    np.add.at(guesses, (inv, predicted), 1)
    np.add.at(correct, (inv, labels), labels == predicted)
    draws = np.random.default_rng(seed).multinomial(n, np.full(n, 1/n), size=samples)
    supports, detections, tp = draws @ truth, draws @ guesses, draws @ correct
    f1 = 2 * tp[:, :22] / np.maximum(1, supports[:, :22] + detections[:, :22])
    # Match audit_classifier's macro definition when a young policy has not
    # observed all 22 achievements. Keep this class set fixed across draws.
    supported = truth.sum(0)[:22] > 0
    macro = f1[:, supported].mean(1)
    return dict(accuracy_95_ci=np.percentile(tp.sum(1)/supports.sum(1), [2.5, 97.5]).tolist(),
                macro_f1_95_ci=np.percentile(macro, [2.5, 97.5]).tolist()), macro, tp.sum(1)/supports.sum(1)


def save_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def save_inference_model(path, centers, method, metadata, grouping):
    # Deliberately does NOT contain achievement IDs, semantic names or mappings.
    np.savez_compressed(path, centers=centers, representation=np.asarray(method),
        run_path=np.asarray(metadata['run_path']), checkpoint=np.asarray(metadata['checkpoint']),
        grouping=np.asarray(grouping), decoder=np.asarray('normalized event delta, nearest Euclidean prototype'),
        health_reward_per_point=np.asarray(.1), achievement_proxy_threshold=np.asarray(.5))


def resolve_encoder_metadata(collection_metadata, run_path=None, checkpoint=None):
    """Keep collection provenance separate from the checkpoint being probed."""
    if (run_path is None) != (checkpoint is None):
        raise ValueError('Supply both encoder run and checkpoint')
    metadata = dict(collection_metadata)
    if run_path is not None:
        run = Path(run_path).resolve()
        if not (run/'files/policies'/str(checkpoint)).is_dir():
            raise FileNotFoundError(f'Encoder checkpoint missing: {run}, {checkpoint}')
        metadata.update(run_path=str(run), checkpoint=int(checkpoint))
    return metadata


def predict_file(model_path, input_path, output_path):
    """Input NPZ: before[N,1345], after[N,1345], rewards[N]; no labels required."""
    model = np.load(model_path, allow_pickle=False)
    data = dict(np.load(input_path, allow_pickle=False))
    for key in ('before', 'after', 'rewards'):
        if key not in data:
            raise ValueError(f'Missing {key}; a static observation alone cannot identify an event')
    if data['before'].shape != data['after'].shape or data['before'].shape[1] != 1345:
        raise ValueError('Expected matching [N,1345] observations')
    proxy = achievement_reward_proxy(data['before'], data['after'], data['rewards'])
    candidate = proxy > float(model['achievement_proxy_threshold'])
    group = np.full(len(proxy), -1, np.int32)
    distances = np.full(len(proxy), np.nan)
    if candidate.any():
        bank = dict(before=data['before'][candidate], after=data['after'][candidate],
                    actions=np.zeros(int(candidate.sum()), np.int32))
        features, _ = embed_bank(bank, dict(run_path=str(model['run_path']),
                                    checkpoint=int(model['checkpoint'])), 512, 50)
        x = normalize(features[str(model['representation'])]); centers = model['centers']
        group[candidate] = nearest_prototypes(x, centers)
        distances[candidate] = ((x - centers[group[candidate]])**2).sum(1)
    if output_path.exists():
        raise RuntimeError(f'Refusing overwrite {output_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_csv(output_path, [dict(transition=i, achievement_reward_proxy=float(proxy[i]),
                              group_id=int(group[i]), squared_distance=float(distances[i])) for i in range(len(proxy))])
    print('Predictions:', output_path)


def write_group_frequencies(out):
    """Label-free rarity counters and event-bank indices for later practice."""
    # Materialize once: repeatedly indexing an NpzFile would decompress a full
    # array for every reference event.
    candidates = dict(np.load(out/'candidate_events.npz', allow_pickle=False))
    reference = np.flatnonzero(candidates['split_id'] == 0)
    uids = candidates['episode_uid'][reference]
    total_episodes = len(np.unique(uids))
    rows = []; events = []
    for path in sorted(out.glob('*_fit.npz')):
        fit = np.load(path, allow_pickle=False)
        group_ids = fit['reference_groups']
        model_name = path.name.replace('_fit.npz', '_model.npz')
        for group in range(len(fit['centers'])):
            selected = np.flatnonzero(group_ids == group)
            episodes = len(np.unique(uids[selected]))
            rows.append(dict(model=model_name, group_id=group, reference_events=len(selected),
                distinct_normal_episodes=episodes, total_normal_episodes=total_episodes,
                normal_episode_frequency=episodes/total_episodes))
        for local, group in enumerate(group_ids):
            if group < 0:
                continue
            index = reference[local]; uid = int(candidates['episode_uid'][index])
            step = int(candidates['steps'][index])
            events.append(dict(model=model_name, group_id=int(group), episode_uid=uid,
                source_dataset=uid//100000, source_episode=uid%100000,
                source_transition_index=int(candidates['source_indices'][index]), pre_event_state_index=step,
                post_event_state_index=step+1, leadup_1=step,
                leadup_5=step-4 if step>=4 else -1, leadup_20=step-19 if step>=19 else -1))
    save_csv(out/'group_frequency.csv', rows)
    save_csv(out/'grouped_reference_events.csv', events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT/'concept_mapping/runs/achievement_clustering_offline_10b_batch2000')
    parser.add_argument('--out', type=Path, default=ROOT/'concept_mapping/runs/transport_vs_kmeans_achievement_groups_10b_batch2000')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--encoder_run', type=Path, help='Optional frozen encoder run instead of the collection policy')
    parser.add_argument('--encoder_checkpoint', type=int, help='Required together with --encoder_run')
    parser.add_argument('--predict_model', type=Path)
    parser.add_argument('--predict_input', type=Path)
    parser.add_argument('--predict_output', type=Path)
    args = parser.parse_args()
    if args.predict_model:
        if not args.predict_input or not args.predict_output:
            parser.error('Prediction requires --predict_input and --predict_output')
        predict_file(args.predict_model, args.predict_input, args.predict_output); return
    out = args.out
    if out.exists() and not args.resume:
        raise RuntimeError(f'Refusing overwrite {out}')
    out.mkdir(parents=True, exist_ok=True); started = time.time()
    def progress(stage, **extra):
        (out/'progress.json').write_text(json.dumps(dict(stage=stage, elapsed_seconds=time.time()-started, **extra), indent=2)+'\n')
    progress('loading')
    bank = dict(np.load(args.cache/'transition_bank.npz', allow_pickle=False))
    proxy = achievement_reward_proxy(bank['before'], bank['after'], bank['rewards'])
    candidate = proxy > .5
    indices = np.flatnonzero(candidate)
    labels = bank['labels'][candidate]; uids = bank['episode_uid'][candidate]
    splits = bank['split_id'][candidate]; steps = bank['steps'][candidate]
    tr = np.flatnonzero(splits == 0); te = np.flatnonzero(splits == 1)
    if set(uids[tr]) & set(uids[te]):
        raise AssertionError('Episode leakage')
    collection_metadata = json.loads((DATASETS[0]/'dataset_metadata.json').read_text())
    metadata = resolve_encoder_metadata(collection_metadata, args.encoder_run, args.encoder_checkpoint)
    # Prevent old fitted groups being reused with different representation weights.
    provenance_path = out/'encoder_provenance.json'
    provenance = dict(run_path=metadata['run_path'], checkpoint=metadata['checkpoint'],
                      source_cache=str(args.cache.resolve()),
                      collection_run_path=collection_metadata['run_path'],
                      collection_checkpoint=collection_metadata['checkpoint'])
    if args.resume and provenance_path.exists():
        if json.loads(provenance_path.read_text()) != provenance:
            raise ValueError('Resume encoder/cache provenance differs; use a new output directory')
    elif args.resume and (out/'summary.json').exists():
        previous = json.loads((out/'summary.json').read_text())
        if previous['checkpoint'] != metadata['checkpoint'] or previous['run_path'] != metadata['run_path']:
            raise ValueError('Resume encoder differs from previous fitted groups')
    provenance_path.write_text(json.dumps(provenance, indent=2)+'\n')
    names = [n.lower() for n in metadata['achievement_names']] + ['health_gain', 'health_loss', 'other', 'compound', 'unknown']
    candidate_data = dict(source_indices=indices, episode_uid=uids, steps=steps,
                          split_id=splits, reward_proxy=proxy[candidate], evaluation_labels=labels)
    candidate_path = out/'candidate_events.npz'
    if args.resume and candidate_path.exists():
        existing = dict(np.load(candidate_path, allow_pickle=False))
        if set(existing) != set(candidate_data) or any(
                not np.array_equal(existing[key], value) for key, value in candidate_data.items()):
            raise ValueError('Resume candidate bank differs; use a new output directory')
    else:
        np.savez_compressed(candidate_path, **candidate_data)
    # Audit selection only: health subtraction uses known reward mechanics, never flags.
    selection_audit = dict(selected_events=len(indices), reference_events=len(tr), test_events=len(te),
        non_achievement_candidates=int((labels >= 22).sum()), missed_achievement_events=int(((bank['labels'] < 22) & ~candidate).sum()),
        reward_proxy_values=np.unique(np.round(proxy, 4)).tolist())
    if args.encoder_run is not None:
        feature_path = out/'encoder_features.npz'
        validation_path = out/'encoder_validation.json'
        if args.resume and feature_path.exists() and validation_path.exists():
            candidate_features = dict(np.load(feature_path, allow_pickle=False))
            encoder_validation = json.loads(validation_path.read_text())
        else:
            progress('encoding', encoder_checkpoint=metadata['checkpoint'], candidate_events=len(indices))
            event_bank = {key: bank[key][candidate] for key in ('before', 'after', 'actions')}
            all_features, encoder_validation = embed_bank(event_bank, metadata, 512, 50)
            candidate_features = {key: all_features[key] for key in ('critic_delta', 'actor_delta')}
            np.savez_compressed(feature_path, **candidate_features)
            validation_path.write_text(json.dumps(encoder_validation, indent=2)+'\n')
            del event_bank, all_features
        archive = None
    else:
        candidate_features = None; encoder_validation = None
        archive = np.load(args.cache/'features.npz', allow_pickle=False)
    del bank, proxy
    summaries = []; per_class = []; episode_matches = []; predictions = {}
    for method in ('critic_delta', 'actor_delta'):
        features = candidate_features[method] if candidate_features is not None else archive[method][candidate]
        if len(features) != len(indices):
            raise ValueError('Encoder feature cache does not match candidate events')
        x = normalize(features); reference = x[tr]; query = x[te]
        for seed in (50, 51, 52):
            for grouping in GROUPINGS:
                progress('fitting', method=method, seed=seed, grouping=grouping)
                prefix = f'{method}_{grouping}_seed{seed}'
                artifact = out/f'{prefix}_fit.npz'
                if args.resume and artifact.exists():
                    fitted = dict(np.load(artifact, allow_pickle=False))
                elif grouping == 'transport_anchor':
                    centers, reference_groups, confidence, anchor, uid, residual, ref_audit = build_transport_groups(reference, uids[tr], seed, args.workers)
                    context_groups, context_confidence, test_residual, test_audit = match_episodes(query, uids[te], reference[anchor], args.workers)
                    fitted = dict(centers=centers, reference_groups=reference_groups,
                        reference_confidence=confidence, anchor_indices=anchor, anchor_episode_uid=np.asarray(uid),
                        context_groups=context_groups, context_confidence=context_confidence,
                        max_capacity_violation=np.asarray(max(residual, test_residual)),
                        **{f'reference_{name}': value for name, value in ref_audit.items()},
                        **{f'test_{name}': value for name, value in test_audit.items()})
                else:
                    k = int(grouping.replace('kmeans', ''))
                    centers = fit_kmeans_unlabeled(reference, k, seed, iterations=25)
                    fitted = dict(centers=centers, reference_groups=nearest_prototypes(reference, centers))
                if not (args.resume and artifact.exists()):
                    np.savez_compressed(artifact, **fitted)
                centers = fitted['centers']; reference_groups = fitted['reference_groups']; k = len(centers)
                groups = nearest_prototypes(query, centers)
                model_path = out/f'{prefix}_model.npz'
                if not (args.resume and model_path.exists()):
                    save_inference_model(model_path, centers, method, metadata, grouping)
                for decoder, query_groups in [('single_event_prototype', groups)] + (
                        [('whole_episode_transport', fitted['context_groups'])] if grouping == 'transport_anchor' else []):
                    result = audit_classifier(labels[tr], reference_groups, labels[te], query_groups, k)
                    metrics, precision, recall, f1, support, mapping, predicted, table = result
                    key = dict(representation=method, grouping=grouping, seed=seed, decoder=decoder)
                    row = dict(**key, groups=k, reference_n=len(tr), test_n=len(te),
                        anchor_episode_uid=int(fitted['anchor_episode_uid']) if grouping == 'transport_anchor' else -1,
                        conflicting_reference_episode_matches=int(fitted['reference_conflict_episodes']) if grouping == 'transport_anchor' else 0,
                        conflicting_test_episode_matches=int(fitted['test_conflict_episodes']) if grouping == 'transport_anchor' else 0,
                        max_transport_capacity_violation=float(fitted['max_capacity_violation']) if grouping == 'transport_anchor' else 0., **metrics)
                    summaries.append(row); predictions[(method, grouping, seed, decoder)] = predicted
                    for c in range(22):
                        per_class.append(dict(**key, achievement=names[c], test_n=int(support[c]),
                            precision=float(precision[c]), recall=float(recall[c]), f1=float(f1[c]),
                            reference_groups=int((mapping == c).sum())))
                    event_path = out/f'{prefix}_{decoder}_per_event.csv'
                    if not (args.resume and event_path.exists()):
                        save_csv(event_path, [dict(source_event=int(indices[i]),
                            episode_uid=int(uids[i]), step=int(steps[i]), true_achievement=names[labels[i]],
                            group_id=int(query_groups[j]), predicted_achievement=names[predicted[j]], correct=bool(predicted[j] == labels[i]))
                            for j, i in enumerate(te)])
                    # Semantic group names are saved ONLY in audit artifacts, not inference models.
                    audit_path = out/f'{prefix}_group_audit.csv'
                    if not (args.resume and audit_path.exists()):
                        save_csv(audit_path, [dict(group_id=g, semantic_name=names[mapping[g]],
                            reference_n=int(table[:, g].sum()), majority_fraction=float(table[:, g].max()/max(1, table[:, g].sum())))
                            for g in range(k)])
                    print(prefix, decoder, {m: round(metrics[m], 4) for m in ('accuracy', 'macro_f1', 'assignment_coverage')}, flush=True)
                if grouping == 'transport_anchor':
                    anchor = fitted['anchor_indices']; anchor_labels = labels[tr][anchor]
                    context_groups = fitted['context_groups']
                    for episode in np.unique(uids[te]):
                        local = np.flatnonzero(uids[te] == episode)
                        accepted = context_groups[local] >= 0
                        correct = int((anchor_labels[context_groups[local][accepted]] == labels[te][local][accepted]).sum())
                        shared = len(set(anchor_labels.tolist()) & set(labels[te][local].tolist()))
                        raw_mask = uids[te][fitted['test_raw_source']] == episode
                        raw_labels = labels[te][fitted['test_raw_source'][raw_mask]]
                        raw_target_labels = anchor_labels[fitted['test_raw_target'][raw_mask]]
                        episode_matches.append(dict(representation=method, seed=seed, episode_uid=int(episode),
                            accepted_matches=int(accepted.sum()), correct_matches=correct, possible_correct_matches=shared,
                            raw_ad_accepted_matches=int(raw_mask.sum()), raw_ad_correct_matches=int((raw_labels == raw_target_labels).sum()),
                            confidence_mean=float(fitted['context_confidence'][local].mean())))
                save_csv(out/'summary.csv', summaries)
    save_csv(out/'per_achievement.csv', per_class)
    save_csv(out/'episode_matching.csv', episode_matches)
    write_group_frequencies(out)
    # Same episode draws across all methods; conditional on fixed discovery bank/seed50.
    bootstraps = []; draws = {}
    for key, predicted in predictions.items():
        method, grouping, seed, decoder = key
        if seed != 50:
            continue
        interval, f1_draws, accuracy_draws = bootstrap_metrics(labels[te], predicted, uids[te])
        bootstraps.append(dict(representation=method, grouping=grouping, seed=seed, decoder=decoder, **interval))
        draws[key] = (f1_draws, accuracy_draws)
    comparisons = []
    for method in ('critic_delta', 'actor_delta'):
        a = draws[(method, 'transport_anchor', 50, 'single_event_prototype')]
        for grouping in (g for g in GROUPINGS if g.startswith('kmeans')):
            b = draws[(method, grouping, 50, 'single_event_prototype')]
            comparisons.append(dict(representation=method, comparator=grouping,
                macro_f1_difference_95_ci=np.percentile(a[0]-b[0], [2.5, 97.5]).tolist(),
                accuracy_difference_95_ci=np.percentile(a[1]-b[1], [2.5, 97.5]).tolist()))
    payload = dict(selection_audit=selection_audit, checkpoint=metadata['checkpoint'], run_path=metadata['run_path'],
        collection_checkpoint=collection_metadata['checkpoint'], collection_run_path=collection_metadata['run_path'],
        encoder_validation=encoder_validation, frozen_probe_states=True,
        reference_episodes=int(len(np.unique(uids[tr]))), test_episodes=int(len(np.unique(uids[te]))),
        transport_implementation=UPSTREAM, transport_parameters=dict(reg=.05, iterations=100, threshold=.5, unit_event_masses=True),
        grouping_extension='Reference anchor chosen uniformly from maximum-event episodes; raw AD T>.5 matches saved; confidence-sorted greedy one-to-one conflict resolution for persistent groups; means of accepted matches are prototypes.',
        reward_detection='Subtract known .1*health-point-change from net reward, then threshold at .5; no flags.',
        label_usage='Post-fit naming and evaluation only, never discovery or inference models.',
        policy_training=False, auxiliary_training=False, new_rollouts=False, grouping_conditions=list(GROUPINGS),
        results=summaries, bootstrap_intervals=bootstraps, paired_comparisons_seed50=comparisons,
        elapsed_seconds=time.time()-started,
        limitations=['AD-style matching only, not full AD or an AD-trained encoder.',
            'Frozen encoder is tested on fixed states collected by the 10B policy; not evidence of discovery in its own rollouts.',
            'Maximum-event reference episode can miss concepts absent from every anchor.',
            'Kmeans22 explicitly assumes the target number of concepts; 32/64/128 are overclustered controls.',
            'Whole-episode transport sees the event set of a complete held-out episode; not comparable to online single-event inference.',
            'Published 100-iteration POT budget can violate capacities; raw edge diagnostics preserved; greedy conflict repair is our grouping extension.',
            'Saved numerical group models classify reward events from before/after observations, not static observations or future goals.',
            'Health reward decomposition is environment-specific prior knowledge; candidate purity is audited, not inferred.',
            'Initialization seeds are not independent PPO training seeds; bootstrap conditions on fitted reference groups.'])
    (out/'summary.json').write_text(json.dumps(payload, indent=2)+'\n')
    plot_comparison(per_class, names, out, encoder_checkpoint=metadata['checkpoint'])
    progress('complete'); print('Complete:', out, flush=True)


def plot_comparison(rows, names, out, encoder_checkpoint=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold'})
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    if encoder_checkpoint is not None:
        fig.suptitle(f'Frozen PPO encoder: {encoder_checkpoint:,} training steps · fixed probe dataset', fontsize=12)
    for ax, method in zip(axes, ('critic_delta', 'actor_delta')):
        for grouping, color in (('transport_anchor', '#e41a1c'), ('kmeans22', '#2462f2'),
                                ('kmeans32', '#9467bd'),
                                ('kmeans64', '#2ca02c'), ('kmeans128', '#777777')):
            selected = [r for r in rows if r['representation'] == method and r['grouping'] == grouping
                        and r['seed'] == 50 and r['decoder'] == 'single_event_prototype']
            ax.plot(range(22), [100*r['f1'] for r in selected], marker='o', markersize=3,
                    color=color, label=grouping.replace('_', ' '))
        ax.set_title(method.replace('_', ' ') + ': held-out single-event assignment')
        ax.set_ylabel('F1 (%)'); ax.set_ylim(-2, 102); ax.grid(alpha=.2); ax.legend(fontsize=8, ncol=3)
    axes[-1].set_xticks(range(22), names[:22], rotation=55, ha='right', fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .96) if encoder_checkpoint is not None else (0, 0, 1, 1))
    for extension in ('pdf', 'png'):
        fig.savefig(out/f'transport_vs_kmeans.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()

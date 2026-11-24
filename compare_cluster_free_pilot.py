"""Compare frozen normal evaluations, with per-training-seed results retained."""
import argparse
import csv
import json
from pathlib import Path
import re
import numpy as np
from concept_mapping.compare_transport_achievement_groups import save_csv

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'concept_mapping/runs/cluster_free_practice_10m'
CONDITIONS = ('baseline', 'uniform', 'prioritized')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-warmup', action='store_true')
    args = parser.parse_args()
    rows, differences, states = [], [], []
    for seed in (50, 51):
        for condition in CONDITIONS:
            folder = OUT/f'seed{seed}_{condition}'
            progress = folder/'progress.json'
            states.append({**(json.loads(progress.read_text()) if progress.exists() else dict(state='not_started')),
                           'seed': seed, 'condition': condition})
            for budget in (5, 10):
                path = folder/f'normal_evaluation_{budget}m/evaluation.json'
                if not path.exists():
                    continue
                evaluation = json.loads(path.read_text())
                row = dict(seed=seed, condition=condition, budget_millions=budget,
                    checkpoint=evaluation['checkpoint'], episodes=evaluation['episodes'],
                    mean_return=evaluation['mean_return'], median_return=evaluation['median_return'],
                    median_length=evaluation['median_length'])
                row.update({name+'_percent': rate*100 for name, rate in evaluation['achievement_rates'].items()})
                rows.append(row)
        for budget in (5, 10):
            folders = [OUT/f'seed{seed}_{condition}'/f'normal_evaluation_{budget}m' for condition in CONDITIONS]
            if not all((f/'evaluation_episodes.csv').exists() for f in folders):
                continue
            data = [list(csv.DictReader((f/'evaluation_episodes.csv').open())) for f in folders]
            for other, control in [(1, 0), (2, 0), (2, 1)]:
                assert [r['episode_uid'] for r in data[other]] == [r['episode_uid'] for r in data[control]]
                delta = np.array([float(a['return_'])-float(b['return_']) for a, b in zip(data[other], data[control])])
                draws = np.random.default_rng(7050+seed).integers(0, len(delta), (2000, len(delta)))
                interval = np.percentile(delta[draws].mean(1), [2.5, 97.5])
                differences.append(dict(seed=seed, budget_millions=budget, condition=CONDITIONS[other],
                    control=CONDITIONS[control], mean_return_difference=float(delta.mean()),
                    episode_bootstrap_ci_low=float(interval[0]), episode_bootstrap_ci_high=float(interval[1])))
    means = []
    for budget in (5, 10):
        for condition in CONDITIONS:
            samples = [r for r in rows if r['condition'] == condition and r['budget_millions'] == budget]
            if len(samples) != 2:
                continue
            means.append(dict(condition=condition, budget_millions=budget, training_seeds=2,
                **{key: float(np.mean([r[key] for r in samples])) for key in samples[0]
                   if key not in ('seed', 'condition', 'budget_millions', 'checkpoint', 'episodes')}))
    if rows:
        save_csv(OUT/'results_per_seed.csv', rows)
    if means:
        save_csv(OUT/'results_seed_means.csv', means)
    if differences:
        save_csv(OUT/'paired_return_differences.csv', differences)
    summary = dict(progress=states, evaluated_runs=len(rows), results_per_seed=rows, seed_means=means,
        caveat='Episode-bootstrap intervals are conditional on each trained policy pair; two training seeds do not establish robust across-seed uncertainty.')
    if args.verify_warmup:
        import jax
        from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer
        checks = []
        for seed in (50, 51):
            leaves = []
            for condition in CONDITIONS:
                log = OUT/f'seed{seed}_{condition}/training.log'
                if not log.exists():
                    break
                matches = re.findall(r'Run data is saved locally in (\S+)', log.read_text())
                if not matches:
                    break
                manager = CheckpointManager(str(Path(matches[-1])/'files/policies'), PyTreeCheckpointer(), CheckpointManagerOptions(create=False))
                if 1007616 not in manager.all_steps():
                    break
                weights = manager.restore(1007616)['params']
                leaves.append([np.asarray(x) for x in jax.tree_util.tree_leaves(weights)])
            if len(leaves) == 3:
                for i in (1, 2):
                    maximum = max(float(np.max(np.abs(a-b))) for a, b in zip(leaves[0], leaves[i]))
                    checks.append(dict(seed=seed, condition=CONDITIONS[i], checkpoint=1007616,
                        bitwise_identical_to_baseline=all(np.array_equal(a, b) for a, b in zip(leaves[0], leaves[i])),
                        maximum_parameter_difference=maximum))
        (OUT/'warmup_verification.json').write_text(json.dumps(checks, indent=2)+'\n')
        summary['warmup_verification'] = checks
    (OUT/'comparison_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(dict(progress=states, seed_means=means,
        **({'warmup_verification': summary['warmup_verification']} if args.verify_warmup else {})), indent=2), flush=True)


if __name__ == '__main__':
    main()

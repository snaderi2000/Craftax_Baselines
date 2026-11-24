"""Compare two finished Map + MLP continuations on paired held-out worlds."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


def load_evaluation(arm, wait):
    directory = arm / 'map_mlp'
    evaluation = directory / 'evaluation_wallclock'
    summary_file = evaluation / 'summary.json'
    if wait:
        manifest = json.loads((arm / 'launch_manifest.json').read_text())
        status_file = directory / 'progress.json'
        while True:
            if status_file.exists():
                status = json.loads(status_file.read_text())
                if status['state'] == 'complete' and summary_file.exists():
                    break
                if status['state'] in ('failed', 'error', 'complete', 'complete_with_failures', 'wall_clock_complete'):
                    raise RuntimeError(f'Continuation or final evaluation failed for {arm}: {status}')
            if time.time() > manifest['overall_deadline']:
                raise RuntimeError(f'Evaluation deadline passed for {arm}')
            time.sleep(30)
    summary = json.loads(summary_file.read_text())
    plan = json.loads((directory / 'experiment_metadata.json').read_text())
    with (evaluation / 'episodes.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    return rows, summary, plan


def results(rows, cap):
    metrics = {
        'All 22 (%)': 100*np.array([r['all_22']=='True' for r in rows]),
        'Drank water (%)': 100*np.array([int(r['collect_drink']) for r in rows]),
        'Collected diamond (%)': 100*np.array([int(r['collect_diamond']) for r in rows]),
        'Mean original return': np.array([float(r['original_return']) for r in rows]),
    }
    for label, field, achievement in (
        ('First drink', 'first_drink_step', 'collect_drink'),
        ('First diamond', 'first_diamond_step', 'collect_diamond'),
    ):
        times = np.array([int(r[field]) if r[field] else -1 for r in rows])
        if np.any((times > 0) != np.array([int(r[achievement]) for r in rows], dtype=bool)):
            raise ValueError(f'First occurrence and achievement disagree for {achievement}')
        metrics[f'{label} (steps, capped)'] = np.where(times > 0, times, cap)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    control, control_summary, control_plan = load_evaluation(args.study/'control', args.wait)
    global_map, global_summary, global_plan = load_evaluation(args.study/'global', args.wait)
    for key in ('resume_from', 'requested_training_steps', 'seed', 'lr', 'resume_reset_optimizer'):
        if control_plan[key] != global_plan[key]:
            raise ValueError(f'Training settings differ for {key}')
    if control_summary['map_global_readout'] or not global_summary['map_global_readout']:
        raise ValueError('Unexpected global readout settings')
    for key in ('training_steps', 'evaluation_seed', 'temperature', 'max_episode_steps', 'episodes'):
        if control_summary[key] != global_summary[key]:
            raise ValueError(f'Evaluation settings differ for {key}')
    if control_summary['training_steps'] != control_plan['requested_training_steps']:
        raise ValueError('Training halted before the matched target')
    pairs = lambda rows: [(int(row['world_seed']), int(row['attempt'])) for row in rows]
    if pairs(control) != pairs(global_map) or len({seed for seed, _ in pairs(control)}) != len(control):
        raise ValueError('Expected the same ordered rows with one episode per held-out world')
    cap = control_summary['max_episode_steps']
    before, after = results(control, cap), results(global_map, cap)
    rng = np.random.default_rng(302303)
    metrics = {}
    for name, control_values in before.items():
        difference = after[name] - control_values
        bounds = np.percentile(rng.choice(difference, (10000, len(difference))).mean(axis=1), [2.5, 97.5])
        metrics[name] = dict(control=float(control_values.mean()), global_map=float(after[name].mean()),
                             difference=float(difference.mean()), ci95=bounds.tolist())
    report = dict(worlds=len(control), additional_steps=control_summary['training_steps']-control_plan['start_timesteps'],
                  control_checkpoint=control_summary['checkpoint'], global_checkpoint=global_summary['checkpoint'],
                  metrics=metrics)
    (args.study/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Full-map fine-tuning', '',
             f'Both branches trained for {report["additional_steps"]:,} additional steps from the same checkpoint.',
             f'Frozen evaluation pairs {len(control):,} held-out worlds.', '',
             '| Metric | 33×33 control | Full map | Difference (95% CI) |',
             '|---|---:|---:|---:|']
    for name, data in metrics.items():
        low, high = data['ci95']
        lines.append(f'| {name} | {data["control"]:.2f} | {data["global_map"]:.2f} | '
                     f'{data["difference"]:+.2f} [{low:+.2f}, {high:+.2f}] |')
    lines.extend(['', f'Episodes without a drink or diamond count as {cap:,} steps for the corresponding timing metric.',
                  'Intervals use 10,000 paired bootstrap resamples of held-out worlds. Both branches use one training seed.'])
    (args.study/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()

"""Predeclared replication of one frozen recipe, with no additional search.

Only seeds 250--257 enter the primary estimate. Prior search and confirmation
scores select the recipe but never enter this replication's confidence interval.
"""
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from aux_search.recipes import BASELINE, Recipe
from run_auxiliary_search import Search, ROOT, atomic_json

FROZEN = Recipe('event', 16, .04210318)
SEEDS = tuple(range(250, 258))
GLOBAL_CAP = 190000000
CORE_FILES = ('ppo.py', 'aux_prediction.py', 'models/actor_critic.py', 'wrappers.py',
              'aux_search/worker.py', 'aux_search/evaluate.py')


def fingerprints():
    return {name:hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in CORE_FILES}


def prepare(out, source):
    out, source = Path(out).resolve(), Path(source).resolve()
    parent = json.loads((source / 'study.json').read_text())
    previous = json.loads((source / 'results.json').read_text())
    if parent['state'] != 'complete' or previous['confirmation'][FROZEN.id]['recipe'] != FROZEN.to_dict():
        raise ValueError('Replication requires the completed source study and exact frozen recipe')
    if set(SEEDS) & {t['seed'] for t in parent['trials']}:
        raise ValueError('Replication seeds overlap the source study')
    prior = int(previous['interactions_charged'])
    training_reserve = len(SEEDS) * 2 * 1000000
    remaining = GLOBAL_CAP - prior - training_reserve
    per_trial = remaining // (len(SEEDS) * 2) // 2048 * 2048
    if per_trial < 2048:
        raise ValueError('Insufficient remaining budget for all predeclared pairs')
    expected = dict(schema=1, source=str(source), source_results_sha256=hashlib.sha256(
        (source / 'results.json').read_bytes()).hexdigest(), baseline=BASELINE.to_dict(),
        frozen_recipe=FROZEN.to_dict(), training_seeds=list(SEEDS),
        evaluation_seeds={str(seed):4050 + i for i, seed in enumerate(SEEDS)},
        episodes_per_agent=256, training_budget_per_agent=1000000,
        executed_training_steps_per_agent=999424, planned_trials=16,
        prior_interactions_charged=prior, max_combined_interactions=GLOBAL_CAP,
        evaluation_steps_per_trial=per_trial, evaluation_reserve=16 * per_trial,
        primary_metric='Mean paired difference in original-environment evaluation return: auxiliary minus baseline.',
        uncertainty='Two-sided 95% Student-t interval over eight training-seed paired differences (df=7).',
        previous_seeds_in_primary_estimate=False,
        stopping_rule='Run all eight predeclared pairs; no efficacy-based early stopping, mutations, or seed replacement.',
        incomplete_rule='Incomplete evaluations produce no score; report incomplete replication rather than dropping a seed.',
        core_source_sha256=fingerprints())
    out.mkdir(parents=True, exist_ok=True)
    plan = out / 'protocol.json'
    if plan.exists():
        stored = json.loads(plan.read_text())
        if {k:stored[k] for k in expected} != expected:
            raise ValueError('Frozen protocol differs from this request or source code')
    else:
        atomic_json(plan, dict(**expected, declared_at=time.time()))
    path = out / 'study.json'
    if not path.exists():
        data = dict(schema=1, search_seed=2026, state='prepared', planned_trials=16,
            training_budget_per_agent=1000000, executed_training_steps_per_agent=999424,
            screening_seeds=list(SEEDS), confirmation_seeds=list(SEEDS),
            evaluation_seeds=expected['evaluation_seeds'], evaluation_steps_per_trial=per_trial,
            screening_episodes=256, confirmation_episodes=256,
            max_total_interactions=GLOBAL_CAP, max_evaluation_interactions=16 * per_trial,
            reserved_validation_interactions=0, prior_interactions_charged=prior,
            source=str(source), weight_inheritance=False,
            baseline=dict(id=BASELINE.id, recipe=BASELINE.to_dict()),
            candidates=[dict(id=FROZEN.id, recipe=FROZEN.to_dict(), controller='fixed',
                             generation=0, parent=None, mutation=None)],
            trials=[], populations=[], decisions=[])
        atomic_json(path, data)
    return expected


def paired_summary(baseline, auxiliary):
    from scipy.stats import t
    if len(baseline) != len(auxiliary) or len(baseline) < 2:
        raise ValueError('At least two complete matched seed pairs are required')
    if not all(math.isfinite(x) for x in list(baseline) + list(auxiliary)):
        raise ValueError('Nonfinite return')
    delta = [a - b for b, a in zip(baseline, auxiliary)]
    mean = statistics.mean(delta)
    sd = statistics.stdev(delta)
    error = float(t.ppf(.975, len(delta) - 1)) * sd / math.sqrt(len(delta))
    ci = [mean - error, mean + error]
    return dict(seed_pairs=len(delta), baseline_mean=statistics.mean(baseline),
        auxiliary_mean=statistics.mean(auxiliary), paired_differences=delta,
        mean_difference=mean, paired_standard_deviation=sd, paired_standard_error=sd / math.sqrt(len(delta)),
        difference_ci95=ci, relative_improvement_pct=100 * mean / statistics.mean(baseline),
        positive_seed_pairs=sum(x > 0 for x in delta),
        conclusion='positive' if ci[0] > 0 else 'negative' if ci[1] < 0 else 'inconclusive')


class Replication(Search):
    def __init__(self, out, source, gpus, max_hours=1):
        self.protocol = prepare(out, source)
        super().__init__(out, gpus[0], 2026, gpus=gpus, max_hours=max_hours)

    def interactions_charged(self):
        return self.data.get('prior_interactions_charged', 0) + super().interactions_charged()

    def run(self):
        if fingerprints() != self.protocol['core_source_sha256']:
            raise RuntimeError('Training/evaluation code changed after protocol declaration')
        jobs = []
        for i, seed in enumerate(SEEDS):
            arms = (BASELINE, FROZEN) if i % 2 == 0 else (FROZEN, BASELINE)
            jobs.extend((recipe, seed, True) for recipe in arms)
        self._run_jobs(jobs)
        report = self.report()
        self.data['state'] = 'complete'; self.save()
        print(json.dumps(report['primary'], indent=2), flush=True)
        return report

    def report(self):
        rows, baseline, auxiliary = [], [], []
        achievements = []
        for seed in SEEDS:
            pair = []
            for recipe in (BASELINE, FROZEN):
                trial = self._trial(recipe, seed)
                if trial is None or trial['state'] != 'complete':
                    raise RuntimeError('All predeclared seed pairs must complete before the primary report')
                result = json.loads((self.out / 'trials' / trial['id'] / 'result.json').read_text())
                if (not result['complete'] or result['training_seed'] != seed
                        or result['training_steps'] != 999424 or result['checkpoint'] != 999424
                        or result['episodes'] != 256 or result['recipe'] != recipe.to_dict()
                        or result['evaluation_seed'] != self.protocol['evaluation_seeds'][str(seed)]):
                    raise RuntimeError('Trial differs from the frozen replication protocol')
                pair.append(result)
            b, a = pair
            baseline.append(b['mean_return']); auxiliary.append(a['mean_return'])
            rows.append(dict(training_seed=seed, evaluation_seed=b['evaluation_seed'],
                baseline_return=b['mean_return'], auxiliary_return=a['mean_return'],
                paired_difference=a['mean_return'] - b['mean_return'],
                baseline_evaluation_steps=b['environment_steps'], auxiliary_evaluation_steps=a['environment_steps']))
            achievements.append((b['achievements'], a['achievements']))
        names = sorted(achievements[0][0])
        if len(names) != 22 or any(set(b) != set(names) or set(a) != set(names) for b, a in achievements):
            raise RuntimeError('Missing per-achievement diagnostics')
        achievement_rows = [dict(achievement=name,
            baseline_success_pct=statistics.mean(b[name] for b, _ in achievements),
            auxiliary_success_pct=statistics.mean(a[name] for _, a in achievements),
            difference_percentage_points=statistics.mean(a[name] - b[name] for b, a in achievements)) for name in names]
        primary = paired_summary(baseline, auxiliary)
        for filename, table in [('per_seed.csv', rows), ('achievement_comparison.csv', achievement_rows)]:
            with (self.out / filename).open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
        report = dict(protocol=str(self.out / 'protocol.json'), recipe=FROZEN.to_dict(),
            primary=primary, per_seed=rows, achievements=achievement_rows,
            prior_interactions_charged=self.data['prior_interactions_charged'],
            additional_interactions_charged=self.interactions_charged() - self.data['prior_interactions_charged'],
            combined_interactions_charged=self.interactions_charged(),
            old_search_or_confirmation_seeds_pooled=False,
            confidence_interval_unit='Training seed pair; evaluation episodes are not independent training replications.')
        atomic_json(self.out / 'results.json', report)
        return report


def plot_report(out, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows, summary = report['per_seed'], report['primary']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for row in rows:
        axes[0].plot([0, 1], [row['baseline_return'], row['auxiliary_return']], color='#b7c2ce', alpha=.8)
        axes[0].scatter([0, 1], [row['baseline_return'], row['auxiliary_return']], c=['#d08a41', '#287f6e'], s=30, zorder=3)
    axes[0].set_xticks([0, 1], ['PPO baseline', 'Event predictor'])
    axes[0].set_ylabel('Mean evaluation return per training seed')
    axes[0].set_title('Eight fresh matched seed pairs')
    axes[1].axhline(0, color='#83909e', linewidth=1)
    axes[1].scatter(range(8), summary['paired_differences'], color='#287f6e', s=35)
    mean, ci = summary['mean_difference'], summary['difference_ci95']
    axes[1].errorbar([9], [mean], yerr=[[mean - ci[0]], [ci[1] - mean]], fmt='D', color='#1e4358', capsize=5)
    axes[1].set_xticks(list(range(8)) + [9], [str(s) for s in SEEDS] + ['Mean\n95% CI'])
    axes[1].set_ylabel('Auxiliary − baseline return')
    axes[1].set_title('Student-t interval across training-seed differences')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
    fig.suptitle('Frozen event objective · horizon 16 · weight 0.04210318 · 999,424 training steps')
    for extension in ('png', 'pdf'):
        fig.savefig(Path(out) / f'paired_returns.{extension}', dpi=180)
    plt.close(fig)


def run_cli(args):
    source = args.out.resolve()
    out = (args.replication_out or source / 'replication_event16_8seeds').resolve()
    protocol = prepare(out, source)
    if args.prepare:
        print(json.dumps(dict(protocol=str(out / 'protocol.json'), **protocol), indent=2))
        return
    gpus = [int(g) for g in args.gpus.split(',')] if args.gpus else [args.gpu]
    if args.background:
        command = [sys.executable, '-u', str(ROOT / 'run_auxiliary_search.py'), '--replicate',
                   '--out', str(source), '--replication-out', str(out), '--gpus', ','.join(map(str, gpus)),
                   '--max-hours', str(args.max_hours or 1)]
        with (out / 'controller.log').open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(out / 'controller_process.json', dict(pid=process.pid, gpus=gpus, command=command))
        print(json.dumps(dict(pid=process.pid, out=str(out)), indent=2))
        return
    lock = (out / 'controller.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    replication = Replication(out, source, gpus, max_hours=args.max_hours or 1)
    report = replication.run()
    plot_report(out, report)

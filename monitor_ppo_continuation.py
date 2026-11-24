"""Compare paired PPO evaluations and independently check the selected checkpoint."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def read_outcomes(directory):
    directory = Path(directory)
    summary = json.loads((directory / 'summary.json').read_text())
    with (directory / 'episodes.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    names = list(summary['achievement_success_pct'])
    assert len(names) == 22 and len(rows) == summary['episodes']
    outcomes = {}
    for row in rows:
        all_22 = all(int(row[name]) for name in names)
        assert all_22 == (row['all_22'].lower() == 'true')
        key = str(row.get('world_seed', row['episode'])) + ':' + str(row.get('attempt', 0))
        assert key not in outcomes
        outcomes[key] = dict(all_22=all_22, hit_limit=int(row['length']) == 10000,
                             timeout=row['timeout'].lower() == 'true')
    result = dict(checkpoint=summary['checkpoint'], steps=summary['training_steps'],
        episodes=len(rows), all_22=sum(x['all_22'] for x in outcomes.values()),
        reached_10000=sum(x['hit_limit'] for x in outcomes.values()),
        both=sum(x['all_22'] and x['hit_limit'] for x in outcomes.values()),
        survived_to_time_limit=sum(x['timeout'] for x in outcomes.values()),
        lava_deaths=sum(row['lava'].lower() == 'true' for row in rows),
        directory=str(directory.resolve()))
    write_json(directory / 'outcome_counts.json', result)
    return result, outcomes


def compare_outcomes(directory, baseline_directory):
    result, outcomes = read_outcomes(directory)
    _, baseline = read_outcomes(baseline_directory)
    assert outcomes.keys() == baseline.keys(), 'Paired evaluations must use identical worlds/attempts'
    result['recovered_all22_failures'] = sum(not baseline[k]['all_22'] and v['all_22']
                                           for k, v in outcomes.items())
    result['new_all22_failures'] = sum(baseline[k]['all_22'] and not v['all_22']
                                     for k, v in outcomes.items())
    return result


def audit_worlds(path, output, reference=None):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from craftax.craftax_env import make_craftax_env_from_name
    data = json.loads(Path(path).read_text())
    seeds = data['world_seeds'] if isinstance(data, dict) else data
    assert len(seeds) == 100 and len(set(seeds)) == 100
    env = make_craftax_env_from_name('Craftax-Classic-Symbolic-v1', auto_reset=False)
    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(seeds, jnp.uint32))
    _, states = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))(keys, env.default_params)
    maps = np.asarray(jax.device_get(states.map))
    hashes = [hashlib.sha256(value.tobytes()).hexdigest() for value in maps]
    assert len(set(hashes)) == 100
    if reference:
        previous = json.loads(Path(reference).read_text())
        assert not set(hashes).intersection(x['initial_map_sha256'] for x in previous['worlds'])
    write_json(output, dict(passed=True, distinct_seeds=100, distinct_starting_maps=100,
        disjoint_from_reference=bool(reference), worlds=[dict(episode=i, world_seed=seed,
        initial_map_sha256=digest) for i, (seed, digest) in enumerate(zip(seeds, hashes))]))


def evaluate_fresh(study, checkpoint, label, args):
    from cap_reward_shaped_runs import try_evaluation_lock
    from run_ppo_study import require_free
    from run_reward_shaped_architectures import ROOT, PYTHON, environment
    output = study / ('fresh_' + label)
    if (output / 'summary.json').exists():
        return read_outcomes(output)[0]
    lock = None
    deadline = time.monotonic() + 1800
    while lock is None:
        lock = try_evaluation_lock(args.gpu)
        if lock is None:
            if time.monotonic() > deadline:
                raise TimeoutError('Fresh-evaluation GPU stayed reserved for 30 minutes')
            time.sleep(10)
    try:
        require_free(args.gpu)
        env = environment(args.gpu)
        audit = study / 'fresh_unique_map_validation.json'
        if not audit.exists():
            command = [PYTHON, '-u', str(Path(__file__).resolve()), '--audit-worlds',
                str(study / 'fresh_worlds_100.json'), '--audit-out', str(audit)]
            reference = Path(args.baseline).parent / 'unique_map_validation.json'
            if reference.exists():
                command.extend(['--reference-map-audit', str(reference)])
            with (study / 'fresh_map_audit.log').open('w') as stream:
                subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, check=True, timeout=300)
        command = [PYTHON, '-u', str(ROOT / 'evaluate_reward_shaped_ppo.py'),
            '--checkpoint', str(checkpoint), '--config-json', str(study / 'map_mlp/inference_config.json'),
            '--out', str(output), '--episodes', '100', '--batch', '25', '--seed', str(args.seed),
            '--world-seeds-file', str(study / 'fresh_worlds_100.json'), '--repeats', '1',
            '--temperature', '1']
        with (study / ('fresh_' + label + '.log')).open('w') as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True, timeout=600)
    finally:
        lock.close()
    return read_outcomes(output)[0]


def monitor(args):
    study = args.study.resolve()
    plan = json.loads((study / 'continuation_plan.json').read_text())
    baseline, _ = read_outcomes(args.baseline)
    assert baseline['episodes'] == 100
    def status(state, **details):
        write_json(study / 'assessment_status.json', dict(state=state, time=time.time(), **details))
    status('evaluating_fresh_baseline')
    print('Evaluating source checkpoint on a separate 100-world set', flush=True)
    fresh_baseline = evaluate_fresh(study, plan['source_checkpoint'], 'baseline', args)
    records = {}
    last_printed = None
    while True:
        for path in (study / 'map_mlp').glob('evaluation_*/summary.json'):
            if str(path) in records:
                continue
            try:
                record = compare_outcomes(path.parent, args.baseline)
            except (OSError, ValueError):
                continue
            record['goal_met_on_paired_set'] = record['all_22'] >= args.goal
            records[str(path)] = record
            print('Paired evaluation:', json.dumps(record), flush=True)
        ordered = sorted(records.values(), key=lambda x: x['steps'])
        report = dict(goal_all_22_out_of_100=args.goal, paired_baseline=baseline,
            fresh_baseline=fresh_baseline, paired_checkpoint_results=ordered,
            selection_rule='Most all-22 successes on paired set; ties use joint survival and later step.')
        write_json(study / 'evaluation_comparison.json', report)
        if ordered:
            fields = ('steps', 'all_22', 'reached_10000', 'both', 'lava_deaths',
                      'recovered_all22_failures', 'new_all22_failures', 'goal_met_on_paired_set')
            with (study / 'evaluation_comparison.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
                writer.writeheader(); writer.writerows(ordered)
        try:
            progress = json.loads((study / 'map_mlp/progress.json').read_text())
        except (OSError, ValueError):
            time.sleep(10)
            continue
        marker = (progress.get('state'), progress.get('latest_checkpoint'), len(ordered))
        if marker != last_printed:
            print('Training:', json.dumps({k: progress.get(k) for k in
                ('state', 'latest_checkpoint', 'training_exit_code')}), flush=True)
            last_printed = marker
        status('monitoring', latest_checkpoint=progress.get('latest_checkpoint'),
               paired_evaluations_complete=len(ordered), fresh_baseline=fresh_baseline)
        terminal = ('complete', 'complete_with_failures', 'wall_clock_complete', 'failed', 'error')
        if progress.get('state') in terminal:
            if progress.get('training_exit_code') != 0 or not progress.get('final_checkpoint'):
                status('failed', reason='Training did not finish successfully', training_progress=progress)
                raise RuntimeError('Training did not finish with a successful final checkpoint')
            break
        time.sleep(20)
    final_checkpoint = progress['final_checkpoint']
    final_paired = study / 'map_mlp/evaluation_wallclock'
    if not (final_paired / 'summary.json').exists():
        raise RuntimeError('Final paired evaluation missing; inspect the training controller logs')
    final = compare_outcomes(final_paired, args.baseline)
    new_candidates = ordered + [final]
    candidates = [baseline] + new_candidates
    best = max(candidates, key=lambda x: (x['all_22'], x['both'], x['steps']))
    status('evaluating_fresh_final', best_paired_checkpoint=best)
    fresh_final = evaluate_fresh(study, final_checkpoint, 'final', args)
    if best['checkpoint'] == final_checkpoint:
        fresh_best = fresh_final
    elif best['checkpoint'] == baseline['checkpoint']:
        fresh_best = fresh_baseline
    else:
        fresh_best = evaluate_fresh(study, best['checkpoint'], 'best_paired', args)
    report.update(state='complete', final_paired=final, best_paired=best,
        best_new_paired=max(new_candidates, key=lambda x: (x['all_22'], x['both'], x['steps'])),
        fresh_final=fresh_final, fresh_best_paired_checkpoint=fresh_best,
        evaluation_failures=progress.get('evaluation_failures', []),
        goal_met_on_paired_set=best['all_22'] >= args.goal,
        goal_met_on_fresh_set=fresh_best['all_22'] >= args.goal,
        total_training_steps=int(Path(final_checkpoint).name),
        additional_training_steps=int(Path(final_checkpoint).name) - plan['source_steps'])
    write_json(study / 'evaluation_comparison.json', report)
    status('complete', report=str(study / 'evaluation_comparison.json'))
    print('FINAL COMPARISON\n' + json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=59150)
    parser.add_argument('--goal', type=int, default=96)
    parser.add_argument('--audit-worlds', type=Path)
    parser.add_argument('--audit-out', type=Path)
    parser.add_argument('--reference-map-audit', type=Path)
    args = parser.parse_args()
    if args.audit_worlds:
        audit_worlds(args.audit_worlds, args.audit_out, args.reference_map_audit)
        return
    if not args.study or not args.baseline:
        parser.error('--study and --baseline are required for monitoring')
    try:
        monitor(args)
    except Exception as error:
        write_json(args.study / 'assessment_status.json',
                   dict(state='failed', time=time.time(), error=str(error)))
        raise


if __name__ == '__main__':
    main()

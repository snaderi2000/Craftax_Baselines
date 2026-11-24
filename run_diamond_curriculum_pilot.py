"""Matched fresh PPO: normal-only vs late 25% diamond-dense episode curriculum."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'concept_mapping/runs/diamond_density_curriculum_100m'
PYTHON = '/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='50')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--out', default=str(OUT))
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--total_timesteps', type=int, default=100000000)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    gpus = [int(g) for g in args.gpus.split(',')]
    if len(set(seeds)) != len(seeds) or len(set(gpus)) != len(gpus):
        raise ValueError('Training seeds and GPU assignments must be unique')
    if len(gpus) != 2*len(seeds):
        raise ValueError('Assign exactly two GPUs per training seed')
    if args.total_timesteps < 50000000:
        raise ValueError('Budget must reach the 50M curriculum boundary')
    out = Path(args.out).resolve()
    if seeds != [50] and out == OUT:
        raise ValueError('Use a separate --out directory to preserve the original seed-50 pilot')
    os.chdir(ROOT)
    specifications = [(f'seed{seed}_{condition}' if len(seeds) > 1 else condition, seed, condition)
                      for seed in seeds for condition in ['baseline', 'curriculum']]
    names = [name for name, _, _ in specifications]
    if args.dry_run:
        print(json.dumps([dict(name=name, seed=seed, condition=condition, gpu=gpu,
                               output=str(out/name), environment_budget=args.total_timesteps,
                               dense_probability=.25 if condition == 'curriculum' else 0.)
                          for gpu, (name, seed, condition) in zip(gpus, specifications)], indent=2))
        return
    for name in names:
        directory = out/name
        directory.mkdir(parents=True, exist_ok=True)
        if (directory/'command.json').exists():
            raise RuntimeError(f'Refusing to overwrite {directory}')
    original = json.loads((ROOT/'concept_mapping/runs/cppo_100m_pilot/baseline/command.json').read_text())['command']
    common = original[:original.index('--run_name')]
    common[common.index('--total_timesteps')+1] = str(args.total_timesteps)
    if args.total_timesteps > 100000000:
        milestones = common.index('--save_policy_milestones')+1
        common[milestones] += ',100000000,150000000'
    environment = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false',
                       OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       WANDB_INIT_TIMEOUT='300')
    jobs, completed, failures = {}, [], {}
    for gpu, (name, seed, condition) in zip(gpus, specifications):
        probability = '.25' if condition == 'curriculum' else '0'
        seeded_common = common.copy()
        seeded_common[seeded_common.index('--seed')+1] = str(seed)
        command = seeded_common + ['--run_name', f'diamond-density-{condition}-{args.total_timesteps//1000000}m-seed{seed}',
                            '--diamond_curriculum', '--diamond_curriculum_start', '50000000',
                            '--diamond_dense_probability', probability,
                            '--diamond_density_multiplier', '10']
        env = dict(environment, CUDA_VISIBLE_DEVICES=str(gpu))
        directory = out/name
        (directory/'command.json').write_text(json.dumps(dict(command=command, gpu=gpu), indent=2)+'\n')
        with (directory/'training.log').open('w') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        jobs[name] = (process, env, seed, condition)
        print(f'Launched {name} GPU {gpu} PID {process.pid}', flush=True)
    while len(completed)+len(failures) < len(names):
        for name, (process, env, seed, condition) in jobs.items():
            if name in completed or name in failures:
                continue
            code = process.poll()
            if code is None:
                continue
            if code:
                failures[name] = f'training exit {code}'
                continue
            directory = out/name
            matches = re.findall(r'Run data is saved locally in (\S+)', (directory/'training.log').read_text())
            if not matches:
                failures[name] = 'Missing W&B run path'
                continue
            run = matches[-1]
            (directory/'run_manifest.json').write_text(json.dumps(dict(condition=condition,
                local_run=run, wandb_path=f'synaderi-uc-davis/Craftax_Baselines/{run.rsplit("-", 1)[-1]}',
                seed=seed, environment_budget=args.total_timesteps, curriculum_start=50000000,
                dense_probability=.25 if condition == 'curriculum' else 0., density_multiplier=10), indent=2)+'\n')
            with (directory/'evaluation.log').open('w') as log:
                result = subprocess.run([PYTHON, '-u', 'evaluate_diamond_opportunities.py',
                    '--run', run, '--out', str(directory/'normal_evaluation'), '--episodes', '512', '--seed', '2050'],
                    env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                failures[name] = f'evaluation exit {result.returncode}'
            else:
                if args.total_timesteps > 100000000:
                    # First completed PPO update at or above the requested 100M milestone.
                    checkpoint100m = ((100000000+8191)//8192)*8192
                    with (directory/'evaluation_100m.log').open('w') as log:
                        intermediate = subprocess.run([PYTHON, '-u', 'evaluate_diamond_opportunities.py',
                            '--run', run, '--out', str(directory/'normal_evaluation_100m'),
                            '--episodes', '512', '--seed', '2050', '--checkpoint', str(checkpoint100m)],
                            env=env, stdout=log, stderr=subprocess.STDOUT)
                    if intermediate.returncode:
                        failures[name] = f'100M evaluation exit {intermediate.returncode}'
                        continue
                completed.append(name)
                print(f'Completed and evaluated {name} on original normal Craftax', flush=True)
        (out/'progress.json').write_text(json.dumps(dict(completed=completed, failures=failures,
            pids={n:job[0].pid for n,job in jobs.items()}, updated_unix_time=time.time()), indent=2)+'\n')
        if len(completed)+len(failures) < len(names):
            time.sleep(20)
    if failures:
        raise RuntimeError(str(failures))
    results = {n:dict(training_seed=seed, condition=condition,
                     **json.loads((out/n/'normal_evaluation/summary.json').read_text()))
               for n, seed, condition in specifications}
    (out/'comparison.json').write_text(json.dumps(results, indent=2)+'\n')
    if args.total_timesteps > 100000000:
        at100m = {n:dict(training_seed=seed, condition=condition,
                        **json.loads((out/n/'normal_evaluation_100m/summary.json').read_text()))
                  for n, seed, condition in specifications}
        (out/'comparison_100m.json').write_text(json.dumps(at100m, indent=2)+'\n')
    if 50 not in seeds and (OUT/'comparison.json').exists():
        original_results = json.loads((OUT/'comparison.json').read_text())
        combined = {f'seed50_{c}':dict(training_seed=50, condition=c, **r)
                    for c, r in original_results.items()}
        combined.update(at100m if args.total_timesteps > 100000000 else results)
        filename = 'comparison_100m_including_seed50.json' if args.total_timesteps > 100000000 else 'comparison_including_seed50.json'
        (out/filename).write_text(json.dumps(combined, indent=2)+'\n')
    print('Finished: comparison.json is ready.', flush=True)


if __name__ == '__main__':
    main()

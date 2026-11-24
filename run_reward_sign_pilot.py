"""5M reward-sign auxiliary PPO, matched to an existing same-seed baseline."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--gpu', type=int, required=True)
    args = p.parse_args()
    os.chdir(ROOT)
    baseline = ROOT / f'concept_mapping/runs/stone_outcome_aux_5m_seed{args.seed}/baseline'
    out = ROOT / f'concept_mapping/runs/reward_sign_aux_5m_seed{args.seed}'
    if out.exists():
        raise RuntimeError(f'Refusing to overwrite {out}')
    manifest = json.loads((baseline / 'run_manifest.json').read_text())
    baseline_summary = json.loads((baseline / 'evaluation/summary.json').read_text())
    cfg = json.loads((baseline / 'command.json').read_text())['command']
    if (manifest['seed'] != args.seed or manifest['environment_budget'] != 5000000
            or manifest['auxiliary_coefficient'] != 0 or baseline_summary['evaluation_seed'] != 2050
            or cfg[cfg.index('--stone_aux_coef')+1] != '0'):
        raise ValueError('Existing baseline is not matched')
    assert baseline_summary['checkpoint'] == 4997120 and baseline_summary['episodes'] == 256
    free = int(subprocess.check_output(['nvidia-smi','--id',str(args.gpu),
        '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free < 14000:
        raise RuntimeError('Assigned GPU occupied; no other job will be interrupted')
    out.mkdir(parents=True)
    command = cfg.copy()
    command[command.index('--run_name')+1] = f'reward-sign-aux-5m-seed{args.seed}'
    command += ['--reward_sign_aux_coef','.1','--reward_sign_aux_batch','256',
                '--reward_sign_replay_capacity','1024']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), XLA_PYTHON_CLIENT_PREALLOCATE='false',
               OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', WANDB_INIT_TIMEOUT='300')
    def progress(**kwargs):
        (out / 'progress.json').write_text(json.dumps(kwargs,indent=2)+'\n')
    (out / 'command.json').write_text(json.dumps(dict(command=command,gpu=args.gpu,
        baseline=str(baseline),baseline_run=manifest['local_run']),indent=2)+'\n')
    with (out / 'training.log').open('w') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        progress(state='training',seed=args.seed,gpu=args.gpu,pid=process.pid)
        print(f'Training seed {args.seed}, GPU {args.gpu}, PID {process.pid}',flush=True)
        code = process.wait()
    if code:
        progress(state='failed',error=f'training exit {code}')
        raise RuntimeError('Training failed')
    run = re.findall(r'Run data is saved locally in (\S+)',(out / 'training.log').read_text())[-1]
    (out / 'run_manifest.json').write_text(json.dumps(dict(local_run=run,seed=args.seed,
        environment_budget=5000000,gpu=args.gpu,auxiliary_coefficient=.1,evaluation_seed=2050,
        baseline_run=manifest['local_run']),indent=2)+'\n')
    progress(state='evaluating',seed=args.seed,gpu=args.gpu,local_run=run)
    with (out / 'evaluation.log').open('w') as log:
        result = subprocess.run([PYTHON,'-u','evaluate_cppo_pilot.py','--run',run,
            '--out',str(out / 'evaluation'),'--episodes','256','--seed','2050'],env=env,
            stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        progress(state='failed',error='evaluation failed')
        raise RuntimeError('Evaluation failed')
    b = np.asarray([float(r['return_']) for r in csv.DictReader((baseline / 'evaluation/episodes.csv').open())])
    a = np.asarray([float(r['returned_episode_returns']) for r in csv.DictReader((out / 'evaluation/evaluation_episodes.csv').open())])
    assert len(a) == len(b) == 256
    comparison = dict(seed=args.seed,training_steps=4997120,evaluation_episodes_per_policy=256,
        evaluation_seed=2050,baseline_mean_return=float(b.mean()),reward_sign_mean_return=float(a.mean()),
        mean_return_difference=float(a.mean()-b.mean()),baseline_median_return=float(np.median(b)),
        reward_sign_median_return=float(np.median(a)),
        note='Frozen original-environment undiscounted episode returns. Training seeds are the independent replications; evaluation episode ordering is not paired.')
    (out / 'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    progress(state='complete',seed=args.seed,gpu=args.gpu,local_run=run)
    print(json.dumps(comparison,indent=2),flush=True)


if __name__ == '__main__':
    main()

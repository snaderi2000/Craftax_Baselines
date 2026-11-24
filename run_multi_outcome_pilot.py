"""Train 5M multi-outcome PPO and evaluate total return/all achievements."""
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
    p.add_argument('--variant', choices=['multi_outcome','targeted_outcome'], default='multi_outcome')
    args = p.parse_args()
    os.chdir(ROOT)
    previous = ROOT / f'concept_mapping/runs/stone_outcome_aux_5m_seed{args.seed}'
    out = ROOT / f'concept_mapping/runs/{args.variant}_aux_5m_seed{args.seed}'
    if out.exists():
        raise RuntimeError(f'Refusing overwrite: {out}')
    manifests = {name:json.loads((previous/name/'run_manifest.json').read_text())
                 for name in ('baseline','stone_aux')}
    for name, manifest in manifests.items():
        summary = json.loads((previous/name/'evaluation/summary.json').read_text())
        assert manifest['seed'] == args.seed and manifest['environment_budget'] == 5000000
        assert summary['evaluation_seed'] == 2050 and summary['episodes'] == 256 and summary['checkpoint'] == 4997120
        assert manifest['auxiliary_coefficient'] == (0 if name == 'baseline' else .1)
    command = json.loads((previous/'baseline/command.json').read_text())['command']
    assert command[command.index('--stone_aux_coef')+1] == '0'
    command[command.index('--run_name')+1] = f'{args.variant.replace("_","-")}-aux-5m-seed{args.seed}'
    command += [f'--{args.variant}_aux_coef','.1',f'--{args.variant}_aux_batch','256',
                f'--{args.variant}_replay_capacity','1024']
    free = int(subprocess.check_output(['nvidia-smi','--id',str(args.gpu),
        '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if free < 14000:
        raise RuntimeError('GPU occupied; no existing job will be interrupted')
    out.mkdir(parents=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), XLA_PYTHON_CLIENT_PREALLOCATE='false',
               OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', WANDB_INIT_TIMEOUT='300')
    def progress(**status):
        (out/'progress.json').write_text(json.dumps(status,indent=2)+'\n')
    (out/'command.json').write_text(json.dumps(dict(command=command,gpu=args.gpu,
        comparison_runs=manifests),indent=2)+'\n')
    with (out/'training.log').open('w') as log:
        process = subprocess.Popen(command, env=env, stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        progress(state='training',seed=args.seed,gpu=args.gpu,pid=process.pid)
        print(f'Training seed {args.seed}, GPU {args.gpu}, PID {process.pid}',flush=True)
        code = process.wait()
    if code:
        progress(state='failed',error=f'training exit {code}')
        raise RuntimeError('Training failed')
    run = re.findall(r'Run data is saved locally in (\S+)',(out/'training.log').read_text())[-1]
    (out/'run_manifest.json').write_text(json.dumps(dict(local_run=run,seed=args.seed,
        environment_budget=5000000,gpu=args.gpu,auxiliary_coefficient=.1,evaluation_seed=2050),indent=2)+'\n')
    runs = {name:manifest['local_run'] for name,manifest in manifests.items()}
    runs[args.variant] = run
    summaries = {}
    for name, local_run in runs.items():
        progress(state='evaluating',condition=name,seed=args.seed,gpu=args.gpu,local_run=run)
        with (out/f'{name}_evaluation.log').open('w') as log:
            result = subprocess.run([PYTHON,'-u','evaluate_cppo_pilot.py','--run',local_run,
                '--out',str(out/name/'evaluation'),'--episodes','256','--seed','2050'],
                env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            progress(state='failed',condition=name,error='evaluation failed')
            raise RuntimeError('Evaluation failed')
        summary = json.loads((out/name/'evaluation/evaluation.json').read_text())
        assert summary['checkpoint'] == 4997120 and summary['episodes'] == 256
        returns = [float(r['returned_episode_returns']) for r in
                   csv.DictReader((out/name/'evaluation/evaluation_episodes.csv').open())]
        summaries[name] = dict(mean_total_return=summary['returned_episode_returns'],
            median_total_return=float(np.median(returns)),mean_length=summary['returned_episode_lengths'],
            achievement_success_pct={k.split('/',1)[1]:float(np.clip(v,0,100))
                                     for k,v in summary.items() if k.startswith('Achievements/')})
        assert len(summaries[name]['achievement_success_pct']) == 22
        if name != args.variant:
            expected = json.loads((previous/name/'evaluation/summary.json').read_text())['mean_episode_return']
            if not np.isclose(summaries[name]['mean_total_return'],expected,atol=1e-5,rtol=1e-5):
                raise RuntimeError('Control evaluation differs from its previously saved total return')
    comparison = dict(seed=args.seed,training_steps=4997120,evaluation_episodes_per_policy=256,
        evaluation_seed=2050,conditions=summaries,
        note='Original-environment frozen-policy evaluation; success means achievement earned at least once per episode. Training seeds are independent replications.')
    (out/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    with (out/'achievement_comparison.csv').open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['achievement','baseline_success_pct','stone_aux_success_pct',args.variant+'_success_pct'])
        for achievement in summaries['baseline']['achievement_success_pct']:
            writer.writerow([achievement]+[summaries[name]['achievement_success_pct'][achievement] for name in runs])
    progress(state='complete',seed=args.seed,gpu=args.gpu,local_run=run)
    print(json.dumps(comparison,indent=2),flush=True)
    subprocess.run([PYTHON,'compare_multi_outcome_pilot.py','--variant',args.variant],check=True)


if __name__ == '__main__':
    main()

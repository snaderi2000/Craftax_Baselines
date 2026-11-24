"""One fresh 10M cluster-free pilot run, then frozen 5M/10M evaluations."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python'
EXPERIMENT = ROOT/'concept_mapping/runs/cluster_free_practice_10m'


def command_for(condition, seed, total=10000000, capacity=2048):
    return [PYTHON, '-u', str(ROOT/'ppo.py'), '--env_name', 'Craftax-Classic-Symbolic-v1',
        '--num_envs', '128', '--total_timesteps', str(total), '--num_steps', '64', '--update_epochs', '4',
        '--num_minibatches', '8', '--gamma', '.99', '--gae_lambda', '.8', '--clip_eps', '.2',
        '--vf_coef', '.5', '--ent_coef', '.01', '--lr', '.0002', '--max_grad_norm', '1.', '--anneal_lr',
        '--seed', str(seed), '--save_policy', '--save_policy_milestones',
        '65536,983040,1000000,2000000,5000000,7500000',
        '--no-use_optimistic_resets', '--cluster_free_practice', condition, '--cf_capacity', str(capacity),
        '--cf_warmup', '1007616', '--cf_horizon', '30', '--cf_max_reuse', '30',
        '--run_name', f'cluster-free-{condition}-{total//1000000}m-seed{seed}',
        '--wandb_project', 'Craftax_Baselines']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=['baseline', 'uniform', 'prioritized'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    out = EXPERIMENT/f'seed{args.seed}_{args.condition}'
    if out.exists():
        raise FileExistsError(out)
    free = int(subprocess.check_output(['nvidia-smi', '--id', str(args.gpu), '--query-gpu=memory.free',
                                       '--format=csv,noheader,nounits'], text=True).strip())
    if free < 14000:
        raise RuntimeError('GPU occupied; no existing jobs will be interrupted')
    out.mkdir(parents=True)
    command = command_for(args.condition, args.seed)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), XLA_PYTHON_CLIENT_PREALLOCATE='false',
        OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', WANDB_INIT_TIMEOUT='300',
        WANDB_MODE='online')
    def status(**values):
        (out/'progress.json').write_text(json.dumps(dict(seed=args.seed, condition=args.condition,
            gpu=args.gpu, timestamp=time.time(), **values), indent=2)+'\n')
    (out/'command.json').write_text(json.dumps(dict(command=command, gpu=args.gpu), indent=2)+'\n')
    with (out/'training.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        status(state='training', pid=process.pid)
        code = process.wait()
    if code:
        status(state='failed', error=f'training exited {code}')
        raise RuntimeError('Training failed; see training.log')
    matches = re.findall(r'Run data is saved locally in (\S+)', (out/'training.log').read_text())
    run = Path(matches[-1])
    if not run.is_absolute():
        run = ROOT/run
    steps = sorted(int(p.name) for p in (run/'files/policies').iterdir() if p.name.isdigit())
    assert steps[-1] == 10000000//8192*8192
    archive = run/'files/cluster_free_bank'
    manifest = dict(local_run=str(run), seed=args.seed, condition=args.condition, gpu=args.gpu,
        requested_training_steps=10000000, actual_training_steps=steps[-1], checkpoints=steps,
        fresh_training=True, standard_resets=True, warmup_steps=1007616, capacity=2048,
        banking='FIFO; one reservoir-selected pre-reward start per completed normal episode', lag=5,
        practice_lanes=0 if args.condition == 'baseline' else 13, practice_horizon=30, max_reuse=30,
        priority='absolute bootstrapped drill-return error; EMA .2; 80% error-rank + 20% uniform',
        evaluation_seed=6050, evaluation_episodes=512, bank_summary=json.loads((archive/'bank_summary.json').read_text()),
        wandb_path=f'synaderi-uc-davis/Craftax_Baselines/{run.name.rsplit("-",1)[-1]}')
    (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    for budget in (5, 10):
        step = min(s for s in steps if s >= 5000000) if budget == 5 else steps[-1]
        status(state='evaluating', budget_millions=budget, checkpoint=step, local_run=str(run))
        with (out/f'evaluation_{budget}m.log').open('w') as log:
            result = subprocess.run([PYTHON, '-u', str(ROOT/'evaluate_cluster_free_pilot.py'), '--run', str(run),
                '--out', str(out/f'normal_evaluation_{budget}m'), '--checkpoint', str(step)],
                cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status(state='failed', error=f'{budget}M evaluation failed', local_run=str(run))
            raise RuntimeError('Evaluation failed')
    status(state='complete', local_run=str(run))
    print('Completed training and both normal-environment evaluations:', out, flush=True)


if __name__ == '__main__':
    main()

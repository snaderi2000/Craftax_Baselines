"""Fresh ~20M-parameter feedforward PPO; original rewards versus death cost −2.

No architecture changes beyond width. Refuse to start when storage is too low
for the TWO-run logarithmic checkpoint schedule. Never prune existing data.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python'
STUDY = ROOT/'concept_mapping/runs/large_feedforward_20m_10b_seed50'
WIDTH = 1920
PARAMETERS = 19956498
BUDGET = 10000000000
MILESTONES = (65536,100000,150000,220000,330000,500000,750000,1000000,1500000,
    2200000,3300000,5000000,7500000,10000000,15000000,22000000,33000000,50000000,
    75000000,100000000,150000000,220000000,330000000,500000000,750000000,1000000000,
    1500000000,2200000000,3300000000,5000000000,7500000000)


def command_for(condition):
    if condition not in ('baseline', 'death_penalty2'):
        raise ValueError(condition)
    penalty = 0 if condition == 'baseline' else 2
    return [PYTHON, '-u', str(ROOT/'ppo.py'), '--env_name', 'Craftax-Classic-Symbolic-v1',
        '--num_envs', '1024', '--total_timesteps', str(BUDGET), '--num_steps', '64',
        '--update_epochs', '4', '--num_minibatches', '8', '--gamma', '.99', '--gae_lambda', '.8',
        '--clip_eps', '.2', '--vf_coef', '.5', '--ent_coef', '.01', '--lr', '.0002',
        '--max_grad_norm', '1.', '--anneal_lr', '--seed', '50', '--layer_size', str(WIDTH),
        '--activation', 'tanh', '--num_repeats', '1', '--use_optimistic_resets',
        '--optimistic_reset_ratio', '16', '--death_penalty', str(penalty), '--save_policy',
        '--save_policy_milestones', ','.join(map(str, MILESTONES)), '--use_wandb',
        '--wandb_project', 'Craftax_Baselines', '--run_name', f'classic-ff20m-{condition}-10b-seed50']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', choices=['baseline', 'death_penalty2'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--prepare-only', action='store_true', help='Write plan/commands without starting training')
    args = parser.parse_args()
    out = STUDY/args.condition
    if (out/'training.log').exists() or (out/'run_manifest.json').exists():
        raise FileExistsError(f'Refusing to overwrite an existing training attempt: {out}')
    out.mkdir(parents=True, exist_ok=True)
    command = command_for(args.condition)
    free = shutil.disk_usage(out).free
    actual_steps = BUDGET//65536*65536
    plan = dict(command=command, gpu=args.gpu, condition=args.condition, seed=50,
        requested_training_steps=BUDGET, actual_training_steps=actual_steps,
        fresh_training=True, recurrent=False, hidden_width=WIDTH, hidden_layers_per_branch=3,
        activation='tanh', shared_actor_critic_parameters=False, observation_dim=1345, actions=17,
        trainable_parameters=PARAMETERS, actor_parameters=9993617, critic_parameters=9962881,
        additional_death_cost=0 if args.condition == 'baseline' else -2,
        diamond_reward=1, other_rewards_unchanged=True, time_limit_only_penalized=False,
        num_envs=1024, num_steps=64, update_epochs=4, num_minibatches=8, gamma=.99, gae_lambda=.8,
        clip_eps=.2, vf_coef=.5, ent_coef=.01, lr=.0002, anneal_lr=True, max_grad_norm=1.,
        use_optimistic_resets=True, optimistic_reset_ratio=16,
        requested_checkpoints=list(MILESTONES),
        actual_checkpoints=sorted({(s+65535)//65536*65536 for s in MILESTONES}|{actual_steps}),
        required_free_disk_gib_before_launch=30, available_disk_gib=free/1024**3,
        storage_note='About 20GB combined for full optimizer+policy checkpoints, plus logs and headroom; do not use volatile RAM storage for long runs.',
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                       for name in ('ppo.py','models/actor_critic.py','wrappers.py')})
    (out/'experiment_metadata.json').write_text(json.dumps(plan, indent=2)+'\n')
    def status(**data):
        (out/'progress.json').write_text(json.dumps(dict(gpu=args.gpu, condition=args.condition, **data), indent=2)+'\n')
    if args.prepare_only:
        status(state='prepared_not_launched', available_disk_gib=free/1024**3,
               storage_ready=free >= 30*1024**3)
        print(f'Prepared {args.condition}: {PARAMETERS:,} parameters; free disk {free/1024**3:.2f} GiB', flush=True)
        return
    if free < 30*1024**3:
        status(state='blocked_storage', available_disk_gib=free/1024**3, required_disk_gib=30)
        raise RuntimeError('Insufficient storage for two large-model runs: free at least 30 GiB before launching. No data was deleted.')
    gpu_free = int(subprocess.check_output(['nvidia-smi', '--id', str(args.gpu),
        '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    if gpu_free < 14000:
        status(state='blocked_gpu', free_gpu_mib=gpu_free)
        raise RuntimeError('Assigned GPU occupied; no existing jobs will be interrupted')
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.gpu),
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'OPENBLAS_NUM_THREADS': '2',
        'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'WANDB_MODE': 'online', 'WANDB_INIT_TIMEOUT': '300'}
    with (out/'training.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        status(state='training', pid=process.pid)
        code = process.wait()
    matches = re.findall(r'Run data is saved locally in (\S+)', (out/'training.log').read_text())
    run = matches[-1] if matches else None
    if code:
        status(state='failed', exit_code=code, local_run=run)
        raise RuntimeError('Training failed; see training.log')
    steps = sorted(int(p.name) for p in (Path(run)/'files/policies').iterdir() if p.name.isdigit())
    if steps[-1] != actual_steps:
        status(state='failed', error='Unexpected final checkpoint', local_run=run)
        raise RuntimeError('Unexpected final checkpoint')
    (out/'run_manifest.json').write_text(json.dumps(dict(**plan, local_run=run, checkpoints=steps,
        wandb_path='synaderi-uc-davis/Craftax_Baselines/'+Path(run).name.rsplit('-',1)[-1]), indent=2)+'\n')
    status(state='complete', local_run=run, final_checkpoint=steps[-1])


if __name__ == '__main__':
    main()

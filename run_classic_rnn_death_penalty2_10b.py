"""Launch one fresh symbolic-GRU death-cost run with persistent local records."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--death_penalty', type=float, default=2.,
                        help='Positive magnitude of the additional cost applied only on death.')
    parser.add_argument('--diamond_reward', type=float, default=1.,
                        help='Total first-diamond achievement reward, replacing the usual +1.')
    args = parser.parse_args()
    if not math.isfinite(args.death_penalty) or args.death_penalty < 0:
        raise ValueError('Death penalty must be finite and nonnegative')
    if not math.isfinite(args.diamond_reward) or args.diamond_reward < 0:
        raise ValueError('Diamond reward must be finite and nonnegative')
    tag = format(args.death_penalty, 'g').replace('.', 'p')
    diamond_tag = format(args.diamond_reward, 'g').replace('.', 'p')
    suffix = '' if args.diamond_reward == 1. else f'_diamond{diamond_tag}'
    out = ROOT/f'concept_mapping/runs/classic_rnn_death_penalty{tag}{suffix}_10b_seed50'
    run_name = f'classic-rnn-death-penalty{tag}{suffix.replace("_", "-")}-10b-seed50'
    if out.exists():
        raise FileExistsError(f'Refusing overwrite {out}')
    free = int(subprocess.check_output(['nvidia-smi', '--id', str(args.gpu),
        '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip())
    if free < 14000:
        raise RuntimeError('GPU occupied; no other jobs will be interrupted')
    out.mkdir(parents=True)
    command = ['bash', str(ROOT/'run_classic_rnn_death_penalty2_10b.sh')]
    plan = dict(command=command, gpu=args.gpu, trainer='ppo_rnn_og.py',
        environment='Craftax-Classic-Symbolic-v1', symbolic_observations=True,
        recurrent_core='512-unit GRU', fresh_training=True, seed=50,
        requested_training_steps=10000000000, actual_training_steps=9999941632,
        num_envs=1024, num_steps=64, update_epochs=4, num_minibatches=8,
        gamma=.99, gae_lambda=.8, clip_eps=.2, vf_coef=.5, ent_coef=.01,
        lr=.0002, anneal_lr=True, max_grad_norm=1., layer_size=512,
        additional_death_cost=-args.death_penalty, diamond_reward=args.diamond_reward,
        additional_first_diamond_bonus=args.diamond_reward-1.,
        lava_penalized=True, time_limit_only_penalized=False,
        use_optimistic_resets=True, optimistic_reset_ratio=16,
        environment_modification='Opt-in ClassicDeathPenaltyWrapper; installed environment unchanged',
        checkpoints='Same logarithmic requests as feedforward death-cost run; rounded up to completed PPO updates; actual final step saved')
    plan.update(run_name=run_name, death_penalty_magnitude=args.death_penalty,
        launch_command=['/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python',
                        str(Path(__file__).resolve()), '--gpu', str(args.gpu),
                        '--death_penalty', str(args.death_penalty), '--diamond_reward', str(args.diamond_reward)],
        environment_overrides={'RNN_DEATH_PENALTY_GPU': str(args.gpu),
                              'RNN_DEATH_PENALTY': str(args.death_penalty),
                              'RNN_DIAMOND_REWARD': str(args.diamond_reward),
                              'RNN_DEATH_PENALTY_RUN_NAME': run_name})
    (out/'experiment_metadata.json').write_text(json.dumps(plan, indent=2)+'\n')
    def progress(**data):
        (out/'progress.json').write_text(json.dumps(dict(gpu=args.gpu, **data), indent=2)+'\n')
    with (out/'training.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT,
            env={**os.environ, 'RNN_DEATH_PENALTY_GPU': str(args.gpu),
                 'RNN_DEATH_PENALTY': str(args.death_penalty), 'RNN_DEATH_PENALTY_RUN_NAME': run_name,
                 'RNN_DIAMOND_REWARD': str(args.diamond_reward)},
            stdout=log, stderr=subprocess.STDOUT)
        progress(state='training', pid=process.pid)
        code = process.wait()
    matches = re.findall(r'Run data is saved locally in (\S+)', (out/'training.log').read_text())
    run = matches[-1] if matches else None
    if code:
        progress(state='failed', exit_code=code, local_run=run)
        raise RuntimeError('Training failed; see training.log')
    steps = sorted(int(p.name) for p in (Path(run)/'files/policies').iterdir() if p.name.isdigit())
    if steps[-1] != plan['actual_training_steps']:
        progress(state='failed', error='Unexpected final checkpoint', local_run=run)
        raise RuntimeError('Unexpected final checkpoint')
    (out/'run_manifest.json').write_text(json.dumps(dict(**plan, local_run=run, checkpoints=steps,
        wandb_path='synaderi-uc-davis/Craftax_Baselines/'+Path(run).name.rsplit('-', 1)[-1]), indent=2)+'\n')
    progress(state='complete', local_run=run, final_checkpoint=steps[-1])


if __name__ == '__main__':
    main()

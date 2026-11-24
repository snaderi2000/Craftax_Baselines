"""One fresh training run and final evaluation; used by the budgeted controller."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aux_search.recipes import Recipe


def training_command(recipe, seed, out, steps=1000000):
    return [sys.executable, '-u', str(ROOT / 'ppo.py'), '--env_name', 'Craftax-Classic-Symbolic-v1',
            '--num_envs', '128', '--total_timesteps', str(steps), '--num_steps', '64',
            '--update_epochs', '4', '--num_minibatches', '8', '--layer_size', '512',
            '--gamma', '.99', '--gae_lambda', '.8', '--clip_eps', '.2', '--vf_coef', '.5',
            '--ent_coef', '.01', '--lr', '.0002', '--max_grad_norm', '1.', '--anneal_lr',
            '--seed', str(seed), '--no-use_wandb', '--no-use_optimistic_resets', '--save_policy',
            '--local_output_dir', str(Path(out).resolve() / 'files'),
            '--aux_task', recipe.task, '--aux_horizon', str(recipe.horizon),
            '--aux_coef', str(recipe.weight), '--aux_batch', '256', '--aux_target_seed', '2026']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--recipe', required=True, help='JSON recipe; contains no credentials')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--steps', type=int, default=1000000)
    parser.add_argument('--episodes', type=int, default=64)
    parser.add_argument('--evaluation-seed', type=int, default=2050)
    parser.add_argument('--evaluation-limit', type=int, required=True)
    args = parser.parse_args()
    recipe = Recipe(**json.loads(args.recipe))
    args.out.mkdir(parents=True, exist_ok=True)
    command = training_command(recipe, args.seed, args.out, args.steps)
    (args.out / 'command.json').write_text(json.dumps(dict(command=command, recipe=recipe.to_dict()), indent=2))
    with (args.out / 'training.log').open('w') as log:
        code = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if code:
        raise SystemExit(f'PPO failed (exit {code}); see {args.out / "training.log"}')
    log = (args.out / 'training.log').read_text()
    executed = int(re.findall(r'New environment steps:\s*(\d+)', log)[-1])
    if executed != args.steps // 8192 * 8192:
        raise RuntimeError('Unexpected training interaction count')
    from aux_search.evaluate import evaluate
    result = evaluate(args.out / 'files/policies', args.out / 'evaluation', args.evaluation_seed,
                      args.episodes, args.evaluation_limit)
    result.update(training_steps=executed, recipe=recipe.to_dict(), training_seed=args.seed)
    (args.out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))

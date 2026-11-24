"""One from-scratch learning-program lifetime, followed by real-return evaluation."""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aux_search.recipes import BASELINE
from aux_search.worker import training_command
from learning_rule.graphs import PPO, Program


def run(args):
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    program = Program.from_dict(json.loads(args.program.read_text())) if args.program else PPO
    command = training_command(BASELINE, args.seed, out, args.steps)
    if args.program:
        command += ['--learning_program', str(args.program.resolve())]
    (out / 'command.json').write_text(json.dumps(dict(command=command, program_id=program.id,
        mode='graph' if args.program else 'native_ppo'), indent=2))
    with (out / 'training.log').open('w') as log:
        code = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if code:
        raise RuntimeError(f'Trainer failed with exit {code}; inspect {out / "training.log"}')
    actual = int(re.findall(r'New environment steps:\s*(\d+)', (out / 'training.log').read_text())[-1])
    if actual != args.steps // 8192 * 8192:
        raise RuntimeError('Unexpected training interaction count')
    metrics = [json.loads(line) for line in (out / 'files/metrics.jsonl').read_text().splitlines()]
    checks = ['training/optimized_loss', 'training/value_loss', 'training/policy_loss', 'training/approx_kl']
    valid = len(metrics) == actual // 8192 and all(math.isfinite(row[k]) for row in metrics for k in checks)
    valid = valid and all(math.isfinite(v) for row in metrics for k, v in row.items() if k.startswith('program/'))
    metadata = dict(program_id=program.id, training_seed=args.seed, training_steps=actual,
                    evaluation_seed=args.evaluation_seed, valid=valid)
    if not valid:
        result = dict(metadata, complete=False, mean_return=None, environment_steps=0,
                      reason='Nonfinite training loss, diagnostics, or program memory')
    elif args.no_evaluate:
        result = dict(metadata, complete=True, environment_steps=0)
    else:
        from aux_search.evaluate import evaluate
        result = dict(evaluate(out / 'files/policies', out / 'evaluation', args.evaluation_seed,
                              args.episodes, args.evaluation_limit), **metadata)
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--program', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--steps', type=int, default=1000000)
    parser.add_argument('--episodes', type=int, default=64)
    parser.add_argument('--evaluation-seed', type=int, default=5050)
    parser.add_argument('--evaluation-limit', type=int, default=131072)
    parser.add_argument('--no-evaluate', action='store_true')
    run(parser.parse_args())

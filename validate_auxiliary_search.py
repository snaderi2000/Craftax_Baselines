"""Run small real PPO checks before launching the objective search.

Uses 131,072 interactions by default. Tests all task horizons, baseline vs
zero-weight equivalence, local checkpointing, and a real frozen-policy evaluation.
Writes a success artifact that the search controller can require before launch.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'concept_mapping/runs/auxiliary_search_1m/validation')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), XLA_PYTHON_CLIENT_PREALLOCATE='false',
                      OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    from aux_search.recipes import BASELINE, Recipe
    from aux_search.worker import training_command
    # Repeating plain PPO measures GPU floating-point variation independently.
    cases = [('baseline', BASELINE), ('baseline_repeat', BASELINE), ('zero_weight', Recipe('event', 1, .1)),
             ('inverse', Recipe('inverse', 1, .1)), ('observation_delta', Recipe('observation_delta', 4, .1)),
             ('stats_delta', Recipe('stats_delta', 16, .1)), ('event', Recipe('event', 4, .1)),
             ('event16', Recipe('event', 16, .1))]
    for name, recipe in cases:
        directory = args.out / name
        if (directory / 'files/policies').exists():
            continue
        directory.mkdir(exist_ok=True)
        command = training_command(recipe, 123, directory, 16384)
        if name == 'zero_weight':
            command[command.index('--aux_coef') + 1] = '0'
        (directory / 'command.json').write_text(json.dumps(command, indent=2))
        with (directory / 'training.log').open('w') as log:
            print('Checking', name, flush=True)
            code = subprocess.call(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if code:
            raise RuntimeError(f'{name} failed; see {directory / "training.log"}')
    import jax
    import numpy as np
    from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer
    params = []
    for name in ('baseline', 'baseline_repeat', 'zero_weight'):
        manager = CheckpointManager(str(args.out / name / 'files/policies'), PyTreeCheckpointer(),
                                    CheckpointManagerOptions(create=False))
        params.append(manager.restore(manager.latest_step())['params']['params'])
        manager.close()
    differences = []
    for comparison in params[1:]:
        maximum = 0.
        for i in range(8):
            for first, second in zip(jax.tree_util.tree_leaves(params[0][f'Dense_{i}']),
                                     jax.tree_util.tree_leaves(comparison[f'Dense_{i}'])):
                np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-6)
                maximum = max(maximum, float(np.max(np.abs(np.asarray(first) - np.asarray(second)))))
        differences.append(maximum)
    if differences[1] > max(1e-7, 5 * differences[0]):
        raise AssertionError('Zero-weight change exceeds measured GPU baseline variation')
    from aux_search.evaluate import evaluate
    result = evaluate(args.out / 'baseline/files/policies', args.out / 'evaluation', 2050, 32, 250000)
    if not result['complete']:
        raise RuntimeError('Validation evaluation did not finish')
    artifact = dict(complete=True, zero_weight_equivalent=True,
                    baseline_repeat_max_abs_difference=differences[0], zero_weight_max_abs_difference=differences[1],
                    parameter_tolerance=dict(rtol=1e-6, atol=1e-6), task_checks=[name for name, _ in cases],
                    training_interactions=len(cases) * 16384, evaluation_interactions=result['environment_steps'])
    (args.out / 'validation.json').write_text(json.dumps(artifact, indent=2))
    print(json.dumps(artifact, indent=2), flush=True)

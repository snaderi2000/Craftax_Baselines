"""Queue one density pilot on an explicitly allowed GPU, then evaluate normally."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--multiplier', type=int, default=40)
    p.add_argument('--gpus', default='0,1,2,3,4')
    p.add_argument('--seed', type=int, default=50)
    p.add_argument('--out', required=True)
    p.add_argument('--resume_waiting', action='store_true')
    args = p.parse_args()
    os.chdir(ROOT)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'command.json').exists():
        raise RuntimeError(f'Refusing to overwrite launched pilot: {out}')
    if (out/'progress.json').exists():
        previous = json.loads((out/'progress.json').read_text())
        if not args.resume_waiting or previous.get('state') != 'waiting_for_gpu':
            raise RuntimeError(f'Refusing to overwrite existing pilot: {out}')
    allowed = [int(g) for g in args.gpus.split(',')]
    status = dict(state='waiting_for_gpu', allowed_gpus=allowed, seed=args.seed,
                  density_multiplier=args.multiplier, environment_budget=200000000)
    def progress(**updates):
        status.update(updates, updated_unix_time=time.time())
        (out/'progress.json').write_text(json.dumps(status, indent=2)+'\n')
    progress()
    print(f'Queued {args.multiplier}x pilot; allowed GPUs {allowed}', flush=True)
    while True:
        free = {}
        for gpu in allowed:
            result = subprocess.run(['nvidia-smi', '--id', str(gpu), '--query-gpu=memory.free',
                                      '--format=csv,noheader,nounits'], capture_output=True, text=True)
            if result.returncode == 0:
                free[gpu] = int(result.stdout.strip())
        candidates = [g for g in allowed if free.get(g, 0) >= 14000]
        if candidates:
            gpu = candidates[0]
            break
        progress(free_memory_mib=free)
        time.sleep(30)
    source = ROOT/'concept_mapping/runs/diamond_density_curriculum_200m/replications_seed50/curriculum/command.json'
    command = json.loads(source.read_text())['command']
    command[command.index('--seed')+1] = str(args.seed)
    command[command.index('--diamond_density_multiplier')+1] = str(args.multiplier)
    command[command.index('--run_name')+1] = f'diamond-density-{args.multiplier}x-200m-seed{args.seed}'
    (out/'command.json').write_text(json.dumps(dict(command=command, gpu=gpu), indent=2)+'\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_PREALLOCATE='false',
               OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', WANDB_INIT_TIMEOUT='300')
    with (out/'training.log').open('w') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        progress(state='training', gpu=gpu, pid=process.pid)
        print(f'Launched on GPU {gpu}, PID {process.pid}', flush=True)
        code = process.wait()
    if code:
        progress(state='failed', error=f'training exit {code}')
        raise RuntimeError(f'Training exited {code}')
    matches = re.findall(r'Run data is saved locally in (\S+)', (out/'training.log').read_text())
    if not matches:
        progress(state='failed', error='Missing W&B run path')
        raise RuntimeError('Missing W&B run path')
    run = matches[-1]
    manifest = dict(local_run=run, training_seed=args.seed, gpu=gpu, density_multiplier=args.multiplier,
                    environment_budget=200000000, curriculum_start=50000000, dense_probability=.25,
                    wandb_path=f'synaderi-uc-davis/Craftax_Baselines/{run.rsplit("-", 1)[-1]}')
    (out/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    progress(state='evaluating', local_run=run)
    for budget, checkpoint in [(200, None), (100, 100007936)]:
        evaluation = out/f'normal_evaluation_{budget}m'
        cmd = [PYTHON, '-u', 'evaluate_diamond_opportunities.py', '--run', run,
               '--out', str(evaluation), '--episodes', '512', '--seed', '2050']
        if checkpoint is not None:
            cmd += ['--checkpoint', str(checkpoint)]
        with (out/f'evaluation_{budget}m.log').open('w') as log:
            result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            progress(state='failed', error=f'{budget}M evaluation exit {result.returncode}')
            raise RuntimeError(f'{budget}M evaluation failed')
        results = {f'{args.multiplier}x': json.loads((evaluation/'summary.json').read_text())}
        if args.seed == 50:
            reference = ROOT/'concept_mapping/runs/diamond_density_curriculum_200m/replications_seed50'
            subfolder = 'normal_evaluation' if budget == 200 else 'normal_evaluation_100m'
            for name, label in [('baseline', 'baseline'), ('curriculum', '10x')]:
                results[label] = json.loads((reference/name/subfolder/'summary.json').read_text())
        (out/f'comparison_{budget}m.json').write_text(json.dumps(results, indent=2)+'\n')
    progress(state='complete')
    print('Training and both normal-environment evaluations completed.', flush=True)


if __name__ == '__main__':
    main()

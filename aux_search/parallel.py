"""Single-writer, subprocess-based GPU scheduling for independent PPO trials."""
from collections import deque
from functools import partial
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def process_args(pid):
    try:
        return [s.decode() for s in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if s]
    except FileNotFoundError:
        return []


def process_live(pid):
    try:
        state = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
        return state not in ('Z', 'X')
    except FileNotFoundError:
        return False


class AttachedWorker:
    """Monitor a worker retained across a controller handoff, without restarting it."""
    def __init__(self, pid):
        self.pid = pid

    def poll(self):
        return None if process_live(self.pid) else 0

    def wait(self, timeout=None):
        start = time.monotonic()
        while self.poll() is None:
            if timeout is not None and time.monotonic() - start >= timeout:
                raise subprocess.TimeoutExpired('attached worker', timeout)
            time.sleep(.1)
        return 0


class BatchScheduler:
    """The controller alone owns trial IDs, reservations and the study manifest."""

    def _trial(self, recipe, seed):
        key = f'{recipe.id}_seed{seed}'
        return next((t for t in self.data['trials'] if t['id'] == key), None)

    def _finish_trial(self, trial, code):
        path = self.out / 'trials' / trial['id'] / 'result.json'
        if code or not path.exists():
            trial.update(state='failed', finished_at=time.time())
            self.save()
            raise RuntimeError(f'Trial failed; inspect {path.parent / "worker.log"}')
        result = json.loads(path.read_text())
        trial.update(evaluation_steps=result['environment_steps'], training_steps=result['training_steps'],
                     finished_at=time.time())
        if not result['complete'] or not math.isfinite(result.get('mean_return', float('nan'))):
            trial['state'] = 'failed'; self.save()
            raise RuntimeError(f'Incomplete evaluation in {trial["id"]}; no partial score is used')
        trial.update(state='complete', mean_return=result['mean_return'])
        self.save()
        print(f'  GPU {trial.get("gpu", self.gpu)} {trial["id"]}: return={trial["mean_return"]:.4f}; '
              f'charged {self.interactions_charged():,} interactions', flush=True)

    def _adopt_trial(self, trial):
        pid = trial.get('pid')
        if pid and process_live(pid):
            args = process_args(pid)
            expected = str((self.out / 'trials' / trial['id']).resolve())
            suffix = getattr(self, 'worker_suffix', '/aux_search/worker.py')
            if (not any(a.endswith(suffix) for a in args)
                    or '--out' not in args or args[args.index('--out') + 1] != expected):
                raise RuntimeError(f'PID {pid} does not match the saved worker; refusing adoption')
            gpu = trial.get('gpu', self.gpu)
            if gpu not in self.gpus:
                raise RuntimeError(f'Existing worker GPU {gpu} is absent from the requested pool')
            print(f'Keeping existing worker {pid} on GPU {gpu}: {trial["id"]}', flush=True)
            return gpu, AttachedWorker(pid), trial
        path = self.out / 'trials' / trial['id'] / 'result.json'
        if path.exists():
            self._finish_trial(trial, 0)
            return None
        raise RuntimeError(f'Unfinished trial {trial["id"]} has no live worker or result; its reservation is retained')

    def _spawn_trial(self, recipe, seed, confirmation, gpu):
        self.check_time()
        remaining = self.data['max_evaluation_interactions'] - self.evaluation_charged()
        reservation = min(self.data.get('evaluation_steps_per_trial',
                          1000000 if confirmation else 250000), remaining)
        if reservation < 2048 or self.interactions_charged() + 1000000 + reservation > self.data['max_total_interactions']:
            self.data['state'] = 'budget_exhausted'; self.save()
            raise RuntimeError('Study interaction budget exhausted')
        self._wait_gpu(gpu)
        key = f'{recipe.id}_seed{seed}'
        trial = dict(id=key, recipe_id=recipe.id, seed=seed, state='running', gpu=gpu,
                     training_reserved=1000000, evaluation_reserved=reservation,
                     confirmation=confirmation, started_at=time.time())
        self.data['trials'].append(trial); self.data['state'] = 'running'; self.save()
        directory = self.out / 'trials' / key
        directory.mkdir(parents=True, exist_ok=True)
        root = Path(__file__).resolve().parents[1]
        command = [sys.executable, '-u', str(root / 'aux_search/worker.py'), '--out', str(directory.resolve()),
            '--recipe', json.dumps(recipe.to_dict()), '--seed', str(seed),
            '--episodes', str(self.data['confirmation_episodes'] if confirmation else self.data['screening_episodes']),
            '--evaluation-seed', str(self.data.get('evaluation_seeds', {}).get(str(seed),
                                     3050 if confirmation else 2050)), '--evaluation-limit', str(reservation)]
        if hasattr(self, 'worker_command'):
            command = self.worker_command(recipe, seed, confirmation, directory, reservation)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_PREALLOCATE='false',
                   OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        env.update(getattr(self, 'worker_environment', {}))
        # Bound JAX's host thread pools by dividing the available CPU affinity.
        # This is set before importing JAX in either worker or trainer.
        cpus = sorted(os.sched_getaffinity(0))
        ix, n = self.gpus.index(gpu), len(self.gpus)
        subset = cpus[ix * len(cpus) // n:(ix + 1) * len(cpus) // n] or [cpus[ix % len(cpus)]]
        trial['cpu_affinity'] = subset
        print(f'Training {key} on GPU {gpu}: {recipe.to_dict()} '
              f'({"confirmation" if confirmation else "screen"})', flush=True)
        with (directory / 'worker.log').open('w') as log:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, preexec_fn=partial(os.sched_setaffinity, 0, subset))
        trial['pid'] = process.pid; self.save()
        return gpu, process, trial

    def _run_jobs(self, jobs):
        pending, active, seen = deque(), {}, set()
        for recipe, seed, confirmation in jobs:
            key = f'{recipe.id}_seed{seed}'
            if key in seen:
                continue
            seen.add(key)
            prior = self._trial(recipe, seed)
            if prior and prior['state'] == 'complete':
                continue
            if prior:
                adopted = self._adopt_trial(prior)
                if adopted:
                    gpu, process, trial = adopted
                    if gpu in active:
                        raise RuntimeError(f'Two existing workers claim GPU {gpu}')
                    active[gpu] = (process, trial)
            else:
                pending.append((recipe, seed, confirmation))
        error = None
        cooldown = {}
        try:
            while pending or active:
                # Serialize all budget reservations in this single event loop.
                if error is None:
                    for gpu in self.gpus:
                        if gpu in active or not pending or time.monotonic() < cooldown.get(gpu, 0):
                            continue
                        try:
                            launched = self._spawn_trial(*pending[0], gpu)
                        except self.deadline_exception as exc:
                            error = exc
                            break
                        except RuntimeError as exc:
                            if 'occupied' in str(exc):
                                cooldown[gpu] = time.monotonic() + 10
                                continue
                            error = exc
                            break
                        pending.popleft()
                        _, process, trial = launched
                        active[gpu] = process, trial
                for gpu, (process, trial) in list(active.items()):
                    code = process.poll()
                    deadline = min(trial['started_at'] + self.trial_timeout,
                                   self.data.get('run_deadline_at', float('inf')) - 15)
                    if code is None and time.time() >= deadline:
                        self._stop_group(process)
                        trial.update(state='interrupted', finished_at=time.time())
                        error = (self.deadline_exception('Wall-clock deadline') if self.remaining_seconds() < 30
                                 else RuntimeError(f'Worker timeout: {trial["id"]}'))
                        self.save()
                        del active[gpu]
                    elif code is not None:
                        try:
                            self._finish_trial(trial, code)
                        except RuntimeError as exc:
                            error = exc
                        del active[gpu]
                if error is not None and not active:
                    self.data.update(state='time_limit_reached' if isinstance(error, self.deadline_exception) else 'failed',
                                     stop_reason=str(error))
                    self.save()
                    raise error
                if pending or active:
                    # A busy external job never gets interrupted. Keep polling
                    # other devices while retaining the global time limit.
                    if not active:
                        self.check_time()
                    time.sleep(1)
        except BaseException:
            # A failure in the controller must not leave unmanaged new workers.
            for process, trial in active.values():
                if process.poll() is None:
                    self._stop_group(process)
                    trial.update(state='interrupted', finished_at=time.time())
            self.save()
            raise

    def evaluate_recipe(self, recipe, seed, confirmation=False):
        self._run_jobs([(recipe, seed, confirmation)])
        return self._trial(recipe, seed)['mean_return']

    def prefetch(self, recipes, confirmation=False):
        seeds = self.data['confirmation_seeds'] if confirmation else self.data['screening_seeds']
        self._run_jobs([(recipe, seed, confirmation) for recipe in recipes for seed in seeds])

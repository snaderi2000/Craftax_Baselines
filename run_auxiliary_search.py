"""Resumeable 1M-step objective search; never inherits trained agent weights.

Defaults: eight initial candidates; four rounds of eight mutations and eight
random controls; two screening seeds; three finalists plus PPO evaluated on five
fresh seeds. All training and evaluation interactions are budgeted. Local files
only, no W&B connection. Run --prepare first to inspect the initial recipes.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time

from aux_search.recipes import BASELINE, Recipe, initial_population, mutate, random_recipe, tournament
from aux_search.parallel import BatchScheduler, process_args, process_live

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / 'concept_mapping/runs/auxiliary_search_1m'


class TimeLimitReached(RuntimeError):
    """Normal stop: keep completed results and do not start another agent."""


def stop_worker_group(process):
    """Stop only the worker process group created by this controller."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # The trainer is a child of the worker and shares this dedicated group.
    # Kill any remaining child even if the worker exited on the first signal.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def wait_for_free_gpu(gpu):
    """Allow the previous worker's utilization sample to settle after exit."""
    for attempt in range(6):
        free = subprocess.check_output(['nvidia-smi', '--id', str(gpu),
            '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(',')
        if int(free[0]) >= 14000 and int(free[1]) <= 5:
            return
        if attempt < 5:
            time.sleep(2)
    raise RuntimeError(f'GPU {gpu} is occupied; existing jobs are not interrupted')


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def handoff_controller(out):
    """Replace our controller while leaving its already-running workers intact."""
    out = Path(out).resolve()
    meta_path = out / 'controller_process.json'
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text())
    pid = meta['pid']
    if not process_live(pid):
        return
    argv = process_args(pid)
    expected = str(Path(__file__).resolve())
    if (pid == os.getpid() or expected not in argv or '--out' not in argv
            or Path(argv[argv.index('--out') + 1]).resolve() != out):
        raise RuntimeError('Saved PID is not this study controller; refusing handoff')
    os.kill(pid, signal.SIGSTOP)
    terminated = False
    try:
        data = json.loads((out / 'study.json').read_text())
        children_path = Path(f'/proc/{pid}/task/{pid}/children')
        children = [int(p) for p in children_path.read_text().split()] if children_path.exists() else []
        workers = []
        for child in children:
            args = process_args(child)
            if not any(a.endswith('/aux_search/worker.py') for a in args) or '--out' not in args:
                continue
            directory = Path(args[args.index('--out') + 1]).resolve()
            if directory.parent != out / 'trials':
                raise RuntimeError('Worker path differs from this study')
            trial = next(t for t in data['trials'] if t['id'] == directory.name)
            trial.update(pid=child, gpu=trial.get('gpu', meta.get('gpu', 0)))
            workers.append(dict(pid=child, trial=trial['id'], gpu=trial['gpu']))
        # Workers have their own sessions; signal ONLY the verified controller.
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)
        terminated = True
        for _ in range(100):
            if not process_live(pid):
                break
            time.sleep(.05)
        if process_live(pid):
            raise RuntimeError('Previous controller did not exit; no replacement launched')
        atomic_json(out / 'study.json', data)
        atomic_json(out / 'handoff.json', dict(previous_controller=pid, retained_workers=workers,
                                              at=time.time()))
        print(f'Controller handoff complete; retained {len(workers)} active worker(s).', flush=True)
    finally:
        if not terminated and process_live(pid):
            os.kill(pid, signal.SIGCONT)


def initialize(out, seed=2026):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    path = out / 'study.json'
    if path.exists():
        data = json.loads(path.read_text())
        if data['search_seed'] != seed:
            raise ValueError('Existing search seed differs; use a new output directory')
        return data
    recipes = initial_population(seed)
    data = dict(schema=1, search_seed=seed, state='prepared', training_budget_per_agent=1000000,
                executed_training_steps_per_agent=999424, screening_seeds=[50, 51],
                confirmation_seeds=[150, 151, 152, 153, 154], population_size=8, generations=4,
                screening_episodes=64, confirmation_episodes=256,
                max_total_interactions=190000000, max_evaluation_interactions=20000000,
                reserved_validation_interactions=2000000,
                fitness='Mean original-environment evaluation return after 999424 training steps; averaged over training seeds.',
                mutation='One field: target, adjacent horizon, or weight times 0.5/2 in [0.001,1]; bounded log-uniform weight jitter when discrete neighbors repeat.',
                weight_inheritance=False, baseline=dict(id=BASELINE.id, recipe=BASELINE.to_dict()),
                candidates=[dict(id=r.id, recipe=r.to_dict(),
                    controller='initial', generation=0, parent=None, mutation=None) for r in recipes],
                trials=[], populations=[], decisions=[])
    atomic_json(path, data)
    return data


class Search(BatchScheduler):
    deadline_exception = TimeLimitReached
    _wait_gpu = staticmethod(wait_for_free_gpu)
    _stop_group = staticmethod(stop_worker_group)

    def __init__(self, out, gpu, seed, max_hours=None, trial_timeout=1800, gpus=None):
        self.out, self.gpu = Path(out), gpu
        self.gpus = tuple(gpus if gpus is not None else [gpu])
        if not self.gpus or len(set(self.gpus)) != len(self.gpus) or any(type(g) is not int or g < 0 for g in self.gpus):
            raise ValueError('GPU pool must contain distinct nonnegative indices')
        self.gpu = self.gpus[0]
        self.data = initialize(out, seed)
        self.path = self.out / 'study.json'
        if trial_timeout <= 0 or not math.isfinite(trial_timeout):
            raise ValueError('Trial timeout must be positive and finite')
        self.trial_timeout = trial_timeout
        if max_hours is not None:
            if max_hours <= 0 or not math.isfinite(max_hours):
                raise ValueError('Hours must be positive and finite')
            # Persist the deadline once; resuming does not silently extend it.
            if 'run_deadline_at' not in self.data:
                self.data.update(session_started_at=time.time(), max_hours=max_hours,
                                 run_deadline_at=time.time() + max_hours * 3600)
                self.save()
        self.data['gpu_pool'] = list(self.gpus)
        self.save()

    def remaining_seconds(self):
        deadline = self.data.get('run_deadline_at')
        return deadline - time.time() if deadline is not None else float('inf')

    def check_time(self):
        durations = [t['finished_at'] - t['started_at'] for t in self.data['trials']
                     if t['state'] == 'complete' and 'finished_at' in t]
        # Allow room for the slower of recent trials, compilation and cleanup.
        margin = max(180, max(durations[-8:], default=120) * 1.5 + 60)
        if self.remaining_seconds() < margin:
            self.data.update(state='time_limit_reached',
                             stop_reason='Not enough time remains to finish another full trial')
            self.save()
            raise TimeLimitReached(self.data['stop_reason'])

    def save(self):
        atomic_json(self.path, self.data)
        complete = [t for t in self.data['trials'] if t['state'] == 'complete']
        atomic_json(self.out / 'progress.json', dict(state=self.data['state'],
            completed_trials=len(complete), planned_trials=self.data.get('planned_trials', 166), gpu=self.gpu, gpus=list(self.gpus),
            active_trials=[dict(id=t['id'], gpu=t.get('gpu', self.gpu), pid=t.get('pid'))
                           for t in self.data['trials'] if t['state'] == 'running'],
            interactions_charged=self.interactions_charged(),
            evaluation_interactions=self.evaluation_charged(),
            run_deadline_at=self.data.get('run_deadline_at'),
            remaining_seconds=max(0, self.remaining_seconds()) if 'run_deadline_at' in self.data else None,
            latest_trial=self.data['trials'][-1] if self.data['trials'] else None))
        fitness = {}
        for candidate in self.data['candidates'] + [dict(id=BASELINE.id)]:
            runs = [t for t in complete if t['recipe_id'] == candidate['id'] and not t['confirmation']]
            if len(runs) == len(self.data['screening_seeds']):
                fitness[candidate['id']] = statistics.mean(t['mean_return'] for t in runs)
        atomic_json(self.out / 'leaderboard.json', dict(baseline=fitness.get(BASELINE.id),
                    fitness=fitness, decisions=self.data['decisions']))

    def evaluation_charged(self):
        # Pending/crashed evaluations retain their full reservation. Never
        # silently reuse their budget when resuming an interrupted search.
        return sum(t.get('evaluation_steps', t['evaluation_reserved']) for t in self.data['trials'])

    def interactions_charged(self):
        return sum(t['training_reserved'] for t in self.data['trials']) + self.evaluation_charged() + self.data['reserved_validation_interactions']


    def fitness(self, recipe):
        return statistics.mean(self.evaluate_recipe(recipe, seed) for seed in self.data['screening_seeds'])

    def candidates(self, controller, generation=None):
        return [Recipe(**c['recipe']) for c in self.data['candidates']
                if c['controller'] == controller and (generation is None or c['generation'] == generation)]

    def run(self):
        initial = self.candidates('initial')
        self.prefetch([BASELINE] + initial)
        baseline = self.fitness(BASELINE)
        fitness = {r.id:self.fitness(r) for r in initial}
        population = sorted(initial, key=lambda r:(-fitness[r.id], r.id))
        for generation in range(1, self.data['generations'] + 1):
            children = self.candidates('mutation', generation)
            controls = self.candidates('random', generation)
            if not children:
                rng = random.Random(self.data['search_seed'] + generation * 100003)
                used = {c['id'] for c in self.data['candidates']}
                attempts = 0
                while len(children) < self.data['population_size']:
                    attempts += 1
                    if attempts > 10000:
                        raise RuntimeError('Unique one-field mutations exhausted; no duplicate agent training is charged')
                    parent = tournament(population, fitness, rng)
                    child, change = mutate(parent, rng, field='weight' if attempts > 200 else None,
                                           jitter=attempts > 200)
                    if child.id in used:
                        continue
                    children.append(child); used.add(child.id)
                    self.data['candidates'].append(dict(id=child.id, recipe=child.to_dict(),
                        controller='mutation', generation=generation, parent=parent.id, mutation=change))
                # Independent RNG for the matched random-search branch.
                random_rng = random.Random(self.data['search_seed'] + generation * 100019)
                while len(controls) < self.data['population_size']:
                    child = random_recipe(random_rng)
                    if child.id in used:
                        continue
                    controls.append(child); used.add(child.id)
                    self.data['candidates'].append(dict(id=child.id, recipe=child.to_dict(),
                        controller='random', generation=generation, parent=None, mutation=None))
                self.save()  # Persist recipes and parentage BEFORE any training.
            # Interleave the two branches so a wall-clock stop does not give
            # mutation search an entire generation's compute advantage.
            self.prefetch([r for pair in zip(children, controls) for r in pair])
            for pair in zip(children, controls):
                for candidate in pair:
                    fitness[candidate.id] = self.fitness(candidate)
            population = sorted(population + children, key=lambda r:(-fitness[r.id], r.id))[:self.data['population_size']]
            random_pool = initial + self.candidates('random')
            random_pool = [r for r in random_pool if r.id in fitness]
            decision = dict(generation=generation, population=[r.id for r in population],
                            mutation_best=fitness[population[0].id],
                            random_best=max(fitness[r.id] for r in random_pool), baseline=baseline)
            self.data['decisions'] = [d for d in self.data['decisions'] if d['generation'] != generation] + [decision]
            self.save()
            atomic_json(self.out / 'leaderboard.json', dict(baseline=baseline, fitness=fitness, decisions=self.data['decisions']))
        evolved = population[0]
        random_best = max(initial + self.candidates('random'), key=lambda r:fitness[r.id])
        finalists = [evolved]
        if random_best.id != evolved.id:
            finalists.append(random_best)
        all_candidates = initial + self.candidates('mutation') + self.candidates('random')
        for candidate in sorted(all_candidates, key=lambda r:(-fitness[r.id], r.id)):
            if candidate.id not in {r.id for r in finalists}:
                finalists.append(candidate)
            if len(finalists) == 3:
                break
        confirmation = {}
        self.prefetch(finalists + [BASELINE], confirmation=True)
        for recipe in finalists + [BASELINE]:
            returns = [self.evaluate_recipe(recipe, seed, confirmation=True) for seed in self.data['confirmation_seeds']]
            confirmation[recipe.id] = dict(recipe=recipe.to_dict(), returns=returns,
                mean=statistics.mean(returns), standard_deviation=statistics.stdev(returns))
        self.data['confirmation'] = confirmation
        self.data['state'] = 'complete'; self.save()
        best = max(confirmation, key=lambda key: confirmation[key]['mean'])
        baseline_returns = confirmation[BASELINE.id]['returns']
        for result in confirmation.values():
            result['paired_return_differences_from_baseline'] = [a - b for a, b in zip(result['returns'], baseline_returns)]
            result['mean_difference_from_baseline'] = statistics.mean(result['paired_return_differences_from_baseline'])
        atomic_json(self.out / 'results.json', dict(confirmation=confirmation,
                    mutation_winner=evolved.id, random_winner=random_best.id,
                    baseline=BASELINE.id, best_recipe=best,
                    best_recipe_checkpoints=[str(self.out / 'trials' / f'{best}_seed{seed}' / 'files/policies/999424')
                                             for seed in self.data['confirmation_seeds']],
                    interactions_charged=self.interactions_charged()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--search-seed', type=int, default=2026)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpus', type=str, default=None, help='Comma-separated GPU pool; one independent worker per GPU')
    parser.add_argument('--take-over', action='store_true', help='Hand off this study controller without restarting live workers')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--background', action='store_true', help='Launch the study in a retained process and return its PID')
    parser.add_argument('--max-hours', type=float, default=None, help='Persistent wall-clock limit for this search session')
    parser.add_argument('--trial-timeout', type=float, default=1800, help='Maximum seconds for one training/evaluation worker')
    parser.add_argument('--sanity', action='store_true', help='Check one baseline and one auxiliary 1M trial before overnight launch')
    parser.add_argument('--replicate', action='store_true', help='Run the frozen event-16 objective on eight new matched seed pairs')
    parser.add_argument('--replication-out', type=Path, default=None)
    args = parser.parse_args()
    if args.replicate:
        from aux_search.replication import run_cli
        run_cli(args)
        raise SystemExit(0)
    if args.take_over and not args.prepare:
        handoff_controller(args.out)
    if args.prepare:
        data = initialize(args.out, args.search_seed)
        print(json.dumps(dict(path=str(args.out / 'study.json'), candidates=data['candidates']), indent=2))
    elif args.background:
        args.out.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--out', str(args.out.resolve()),
                   '--search-seed', str(args.search_seed), '--gpu', str(args.gpu),
                   '--trial-timeout', str(args.trial_timeout)]
        if args.max_hours is not None:
            command += ['--max-hours', str(args.max_hours)]
        if args.gpus is not None:
            command += ['--gpus', args.gpus]
        with (args.out / 'controller.log').open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(args.out / 'controller_process.json', dict(pid=process.pid, gpu=args.gpu,
                    gpus=[int(g) for g in args.gpus.split(',')] if args.gpus else [args.gpu], command=command))
        print(json.dumps(dict(pid=process.pid, log=str(args.out / 'controller.log')), indent=2))
    else:
        validation = args.out / 'validation/validation.json'
        if not validation.exists() or not json.loads(validation.read_text()).get('complete'):
            raise RuntimeError('Run validate_auxiliary_search.py successfully before starting this study')
        lock = (args.out / 'controller.lock').open('a')
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('A controller is already running for this study')
        search = Search(args.out, args.gpu, args.search_seed, max_hours=args.max_hours,
                        trial_timeout=args.trial_timeout,
                        gpus=[int(g) for g in args.gpus.split(',')] if args.gpus is not None else None)
        try:
            if args.sanity:
                candidate = next(r for r in search.candidates('initial') if r.task == 'observation_delta')
                scores = {r.id:search.evaluate_recipe(r, search.data['screening_seeds'][0])
                          for r in (BASELINE, candidate)}
                search.data['state'] = 'sanity_passed'; search.save()
                atomic_json(args.out / 'sanity.json', dict(complete=True, scores=scores,
                            auxiliary_recipe=candidate.to_dict(),
                            interactions_charged=search.interactions_charged()))
                print('Sanity check passed. Completed trials will be reused by the overnight search.', flush=True)
            else:
                search.run()
        except TimeLimitReached as exc:
            print(f'Stopped cleanly: {exc}', flush=True)

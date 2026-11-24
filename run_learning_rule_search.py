"""Bounded structural learning-rule evolution, with an eight-GPU validation gate."""
import argparse
import csv
import fcntl
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time

# The controller's synthetic screens/checkpoint comparisons run on CPU. Workers
# explicitly override this before importing JAX and see only their assigned GPU.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
from learning_rule.graphs import PPO, Program, actor_critic_program, describe
from learning_rule.evolution import numerical_screen, offspring
from run_auxiliary_search import ROOT, Search, atomic_json, stop_worker_group, wait_for_free_gpu

DEFAULT_OUT = ROOT / 'concept_mapping/runs/learning_rule_evolution_1m'
SOURCE_FILES = ('ppo.py', 'models/actor_critic.py', 'wrappers.py', 'aux_search/evaluate.py',
    'aux_search/worker.py', 'aux_search/parallel.py', 'aux_search/recipes.py', 'run_auxiliary_search.py',
    'learning_rule/graphs.py', 'learning_rule/runtime.py', 'learning_rule/evolution.py',
    'learning_rule/worker.py', 'run_learning_rule_search.py')
SCREEN_SEEDS = [350, 351]
CONFIRM_SEEDS = list(range(450, 458))


def source_hashes():
    return {f:hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in SOURCE_FILES}


def write_program(out, program):
    path = out / 'programs' / f'{program.id}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, program.to_dict())
    path.with_suffix('.txt').write_text(describe(program))
    return path


def prepare(out):
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    protocol = dict(schema=1, max_total_interactions=75000000,
        max_evaluation_interactions=6000000, reserved_validation_interactions=500000,
        training_budget_per_agent=1000000, executed_training_steps_per_agent=999424,
        initial_programs=8, evolved_programs=7, random_control_programs=7,
        screening_seeds=SCREEN_SEEDS, confirmation_seeds=CONFIRM_SEEDS,
        screening_episodes=64, confirmation_episodes=256,
        evaluation_steps_per_trial=131072,
        evaluation_seeds={str(s):5050 + i for i,s in enumerate(SCREEN_SEEDS)} |
                         {str(s):6050 + i for i,s in enumerate(CONFIRM_SEEDS)},
        selection='Mean original-environment evaluation return at 999424 training steps across both screening seeds.',
        confirmation='Freeze the evolved-branch winner and random-control winner, then compare each to PPO on all eight fresh matched seeds; paired Student-t 95% intervals. Treat both contrasts as exploratory.',
        mutation='Insert, delete, replace, or rewire arithmetic nodes in credit, target, loss, and learning-memory graphs. Two to four edits per child.',
        random_control='Same number of proposals and same edit-count distribution; select initial parents uniformly instead of by measured return.',
        diversity='Reject syntactically duplicate programs and identical synthetic output/gradient signatures, with no environment feedback in this screen.',
        initialization='PPO, unclipped actor-critic, and six synthetic-screened random walks (5..12 structural edits) from these two starting graphs.',
        fixed='Network architecture, Adam, gradient clipping, 4 epochs, 8 minibatches, environment, and original rewards. No auxiliary-task labels, checkpoint warm starts, or additional environment data.',
        boundary_rule='Terminal transitions zero both temporal carry and next-value input before evaluating every evolved credit recurrence.',
        invalid_rule='Numerical divergence is recorded as invalid with no return; every training allocation remains charged. Incomplete evaluations are never scored. System failures stop the study.',
        limitations='This pilot searches scalar credit/loss programs and four scalar memory slots. It does not evolve network architecture, add latent prediction heads, or establish cross-environment generality.',
        stopping_rule='One initial population and one matched evolved/random generation, then fixed fresh-seed confirmation; no extension based on results.',
        prior_studies_pooled=False, source_sha256=source_hashes())
    protocol_path = out / 'protocol.json'
    if protocol_path.exists():
        stored = json.loads(protocol_path.read_text())
        if {k:stored[k] for k in protocol} != protocol:
            raise RuntimeError('Study protocol or source changed; use a new output directory')
    else:
        atomic_json(protocol_path, dict(protocol, declared_at=time.time()))
        for filename in SOURCE_FILES:
            target = out / 'source_snapshot' / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / filename, target)
    study_path = out / 'study.json'
    if study_path.exists(): return json.loads(study_path.read_text())
    print('Generating and numerically screening initial graph programs on CPU...', flush=True)
    rng = random.Random(20261015)
    starts = [PPO, actor_critic_program()]
    records, seen, signatures = [], set(), set()
    for p in starts:
        screen = numerical_screen(p)
        if not screen['valid']: raise RuntimeError('Reference program failed numerical screen')
        seen.add(p.id); signatures.add(screen['signature'])
        records.append(dict(id=p.id, recipe=p.to_dict(), controller='initial', parent=None,
                            mutations=[], numerical_screen=screen, generation=0))
        write_program(out, p)
    while len(records) < 8:
        p, changes, screen = offspring(rng.choice(starts), rng, seen, signatures, edits=rng.randint(5, 12))
        records.append(dict(id=p.id, recipe=p.to_dict(), controller='initial',
            parent=changes[0]['parent'], mutations=changes, numerical_screen=screen, generation=0))
        write_program(out, p)
        print('  Initial program', p.id, 'accepted', flush=True)
    data = dict(protocol, state='prepared', search_seed=20261015,
        planned_trials=68, baseline=dict(id=PPO.id, recipe=PPO.to_dict()),
        candidates=records, trials=[], decisions=[], populations=[], weight_inheritance=False)
    atomic_json(study_path, data)
    return data


def validate_gpu_pipeline(out, gpus):
    """Eight simultaneous short lifetimes; measured native-vs-graph equivalence."""
    import jax
    import numpy as np
    from orbax.checkpoint import CheckpointManager, CheckpointManagerOptions, PyTreeCheckpointer
    path = out / 'validation'; path.mkdir(exist_ok=True)
    artifact_path = path / 'validation.json'
    if artifact_path.exists():
        artifact = json.loads(artifact_path.read_text())
        if artifact.get('complete') and artifact['source_sha256'] == source_hashes(): return artifact
        raise RuntimeError('Validation exists but does not match the current source')
    programs = json.loads((out / 'study.json').read_text())['candidates']
    cases = [('native_ppo', None), ('native_repeat', None), ('graph_ppo', PPO), ('graph_repeat', PPO)]
    cases += [(f'mutant_{i}', Program.from_dict(p['recipe'])) for i,p in enumerate(programs[2:6])]
    pending = list(cases); active = {}; started = time.time()
    try:
        while pending or active:
            for gpu in gpus:
                if gpu in active or not pending: continue
                name, program = pending.pop(0); directory = path / name
                if (directory / 'result.json').exists(): continue
                wait_for_free_gpu(gpu); directory.mkdir(exist_ok=True)
                command = [sys.executable, '-u', str(ROOT / 'learning_rule/worker.py'), '--out', str(directory),
                    '--seed', '314', '--steps', '16384', '--no-evaluate']
                if program is not None: command += ['--program', str(write_program(out, program))]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS='cuda',
                    XLA_PYTHON_CLIENT_PREALLOCATE='false', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
                cpus = sorted(os.sched_getaffinity(0)); index = gpus.index(gpu)
                subset = cpus[index * len(cpus) // len(gpus):(index+1) * len(cpus) // len(gpus)] or cpus[:1]
                with (directory / 'worker.log').open('w') as log:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, preexec_fn=partial(os.sched_setaffinity, 0, subset))
                active[gpu] = process, name
                print(f'Validation {name} on GPU {gpu}', flush=True)
            for gpu, (process, name) in list(active.items()):
                code = process.poll()
                if code is None: continue
                result_path = path / name / 'result.json'
                if code or not result_path.exists(): raise RuntimeError(f'Validation worker failed: {name}')
                result = json.loads(result_path.read_text())
                if result['training_steps'] != 16384:
                    raise RuntimeError(f'Validation interaction count differs: {name}')
                if not result['valid'] and name.startswith(('native', 'graph')):
                    raise RuntimeError(f'Reference validation is nonfinite: {name}')
                print(f'Validation {name} complete', flush=True)
                del active[gpu]
            if time.time() - started > 1800: raise RuntimeError('Validation timed out')
            if pending or active: time.sleep(1)
    finally:
        for process, _ in active.values():
            if process.poll() is None: stop_worker_group(process)
    params = []
    for name in ('native_ppo', 'native_repeat', 'graph_ppo', 'graph_repeat'):
        manager = CheckpointManager(str(path / name / 'files/policies'), PyTreeCheckpointer(), CheckpointManagerOptions(create=False))
        params.append(manager.restore(16384)['params']); manager.close()
    differences = []
    for comparison in params[1:]:
        maximum = 0.
        for a,b in zip(jax.tree_util.tree_leaves(params[0]), jax.tree_util.tree_leaves(comparison)):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
            maximum = max(maximum, float(np.max(np.abs(np.asarray(a) - np.asarray(b)))))
        differences.append(maximum)
    if max(differences[1:]) > max(1e-7, 5*differences[0]):
        raise RuntimeError('Graph PPO differs more than measured native GPU variation')
    artifact = dict(complete=True, source_sha256=source_hashes(), cases=[name for name,_ in cases],
        native_repeat_max_abs_difference=differences[0], graph_ppo_max_abs_difference=differences[1],
        graph_repeat_max_abs_difference=differences[2], training_interactions=8*16384,
        reserved_validation_interactions=500000)
    atomic_json(artifact_path, artifact)
    print(json.dumps(artifact, indent=2), flush=True)
    return artifact


class RuleSearch(Search):
    worker_suffix = '/learning_rule/worker.py'
    worker_environment = dict(JAX_PLATFORMS='cuda')

    def __init__(self, out, gpus, max_hours=2):
        prepare(out)
        super().__init__(out, gpus[0], 20261015, max_hours=max_hours, gpus=gpus)

    def save(self):
        atomic_json(self.path, self.data)
        atomic_json(self.out / 'progress.json', dict(state=self.data['state'],
            completed_trials=sum(t['state']=='complete' for t in self.data['trials']),
            invalid_trials=sum(t['state']=='invalid' for t in self.data['trials']),
            planned_trials=self.data['planned_trials'], gpus=list(self.gpus),
            active_trials=[dict(id=t['id'],gpu=t['gpu'],pid=t.get('pid')) for t in self.data['trials'] if t['state']=='running'],
            interactions_charged=self.interactions_charged(), max_total_interactions=self.data['max_total_interactions'],
            evaluation_interactions=self.evaluation_charged(), remaining_seconds=self.remaining_seconds()))

    def worker_command(self, program, seed, confirmation, directory, reservation):
        return [sys.executable, '-u', str(ROOT / 'learning_rule/worker.py'),
            '--out', str(directory.resolve()), '--program', str(write_program(self.out, program)),
            '--seed', str(seed), '--episodes', str(256 if confirmation else 64),
            '--evaluation-seed', str(self.data['evaluation_seeds'][str(seed)]),
            '--evaluation-limit', str(reservation)]

    def _finish_trial(self, trial, code):
        result_path = self.out / 'trials' / trial['id'] / 'result.json'
        if code == 0 and result_path.exists():
            result = json.loads(result_path.read_text())
            if not result['valid']:
                trial.update(state='invalid', reason=result['reason'], mean_return=None,
                    training_steps=result['training_steps'], evaluation_steps=result['environment_steps'], finished_at=time.time())
                self.save(); print(f'INVALID {trial["id"]}: {result["reason"]}', flush=True); return
        super()._finish_trial(trial, code)

    def score(self, program):
        runs = [self._trial(program, seed) for seed in SCREEN_SEEDS]
        return statistics.mean(t['mean_return'] for t in runs) if all(t and t['state']=='complete' for t in runs) else -1e9

    def programs(self, branch=None):
        return [Program.from_dict(c['recipe']) for c in self.data['candidates'] if branch is None or c['controller']==branch]

    def write_leaderboard(self):
        rows = [dict(id=p.id, controller=next(c['controller'] for c in self.data['candidates'] if c['id']==p.id),
            mean_return=self.score(p) if self.score(p)>-1e8 else None,
            program=str(self.out/'programs'/f'{p.id}.txt')) for p in self.programs()]
        atomic_json(self.out / 'leaderboard.json', sorted(rows, key=lambda r: r['mean_return'] if r['mean_return'] is not None else -1e9, reverse=True))

    def run(self):
        self._run_jobs([(p,s,False) for p in self.programs('initial') for s in SCREEN_SEEDS])
        self.write_leaderboard()
        if not any(c['controller']=='evolved' for c in self.data['candidates']):
            rng = random.Random(20261016); controls_rng = random.Random(20261017)
            initial = self.programs('initial'); valid = [p for p in initial if self.score(p)>-1e8]
            if not valid: raise RuntimeError('No valid initial learning program')
            seen = {p.id for p in self.programs()}
            signatures = {c['numerical_screen']['signature'] for c in self.data['candidates']}
            records = []
            for i in range(7):
                count = rng.randint(2, 4)
                contestants = rng.sample(valid, min(3, len(valid)))
                parent = max(contestants, key=self.score)
                for branch, chosen, source_rng in [('evolved', parent, rng),
                                                  ('random', controls_rng.choice(initial), controls_rng)]:
                    p, changes, screen = offspring(chosen, source_rng, seen, signatures, edits=count)
                    records.append(dict(id=p.id, recipe=p.to_dict(), controller=branch, parent=chosen.id,
                        mutations=changes, numerical_screen=screen, generation=1, proposal=i, edits=count))
                    write_program(self.out, p)
                    print(f'Proposed {branch} {p.id}, parent={chosen.id}, edits={count}', flush=True)
            self.data['candidates'].extend(records)
            self.data['decisions'].append(dict(stage='generation1', initial_fitness={p.id:self.score(p) for p in initial}, at=time.time()))
            self.save()
        # Preserve matched proposal/seed counts between branches and interleave jobs.
        generation = [c for c in self.data['candidates'] if c['generation']==1]
        self._run_jobs([(Program.from_dict(c['recipe']),s,False) for c in generation for s in SCREEN_SEEDS])
        self.write_leaderboard()
        if 'finalists' not in self.data:
            best = {branch:max(self.programs(branch), key=self.score) for branch in ('evolved','random')}
            if any(self.score(p)<-1e8 for p in best.values()): raise RuntimeError('A search branch has no valid program')
            self.data['finalists'] = dict(baseline=PPO.id, **{branch:p.id for branch,p in best.items()})
            self.save()
        lookup = {p.id:p for p in self.programs()}
        finalists = [lookup[self.data['finalists'][branch]] for branch in ('baseline','evolved','random')]
        self._run_jobs([(p,s,True) for s in CONFIRM_SEEDS for p in finalists])
        report = self.report(finalists)
        self.data['state']='complete'; self.save()
        return report

    def report(self, finalists):
        from aux_search.replication import paired_summary
        scores = {p.id:[] for p in finalists}; records=[]; achievements={p.id:[] for p in finalists}
        branches = ['baseline','evolved','random']
        for seed in CONFIRM_SEEDS:
            row=dict(training_seed=seed, evaluation_seed=self.data['evaluation_seeds'][str(seed)])
            for branch,p in zip(branches,finalists):
                trial=self._trial(p,seed)
                if trial['state']!='complete': raise RuntimeError('Every frozen finalist seed must complete; no dropping invalid seeds')
                result=json.loads((self.out/'trials'/trial['id']/'result.json').read_text())
                if (result['training_seed']!=seed or result['evaluation_seed']!=row['evaluation_seed']
                    or result['training_steps']!=999424 or result['checkpoint']!=999424
                    or result['program_id']!=p.id or result['episodes']!=256 or not result['complete']):
                    raise RuntimeError('Confirmation differs from the protocol')
                row[branch+'_return']=result['mean_return']; scores[p.id].append(result['mean_return'])
                achievements[p.id].append(result['achievements'])
            row['evolved_minus_baseline']=row['evolved_return']-row['baseline_return']
            row['random_minus_baseline']=row['random_return']-row['baseline_return']
            records.append(row)
        contrasts={branch:paired_summary(scores[finalists[0].id],scores[p.id]) for branch,p in zip(branches[1:],finalists[1:])}
        contrasts['evolved_minus_random']=paired_summary(scores[finalists[2].id],scores[finalists[1].id])
        names=sorted(achievements[finalists[0].id][0]); rates=[]
        if len(names)!=22: raise RuntimeError('Missing achievement diagnostics')
        for name in names:
            rates.append(dict(achievement=name, **{branch+'_success_pct':statistics.mean(r[name] for r in achievements[p.id]) for branch,p in zip(branches,finalists)}))
        for filename,table in [('per_seed.csv',records),('achievement_comparison.csv',rates)]:
            with (self.out/filename).open('w',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
        report=dict(protocol=str(self.out/'protocol.json'),finalists=self.data['finalists'],
            confirmation=contrasts,per_seed=records,achievements=rates,
            interactions_charged=self.interactions_charged(),evaluation_interactions=self.evaluation_charged(),
            invalid_screening_trials=[t['id'] for t in self.data['trials'] if t['state']=='invalid'],
            exploratory=True,old_studies_pooled=False,source_sha256=source_hashes())
        atomic_json(self.out/'results.json',report)
        print(json.dumps(dict(finalists=report['finalists'],confirmation=contrasts,
            interactions_charged=report['interactions_charged']),indent=2),flush=True)
        plot_report(self.out,report)
        return report


def plot_report(out,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for row in report['per_seed']:
        axes[0].plot(range(3),[row[b+'_return'] for b in ('baseline','evolved','random')],color='#c4cbd2',alpha=.8)
        axes[0].scatter(range(3),[row[b+'_return'] for b in ('baseline','evolved','random')],c=['#c78a43','#247f6c','#5775aa'],s=25,zorder=3)
    axes[0].set_xticks(range(3),['PPO','Evolved program','Random-search program'])
    axes[0].set_ylabel('Evaluation return per training seed')
    axes[0].set_title('Eight fresh matched seed triples')
    axes[1].axhline(0,color='#99a3ac')
    for i,branch in enumerate(('evolved','random')):
        summary=report['confirmation'][branch]; mean=summary['mean_difference']; ci=summary['difference_ci95']
        axes[1].scatter([i]*8,summary['paired_differences'],alpha=.5,s=20)
        axes[1].errorbar(i+.12,mean,yerr=[[mean-ci[0]],[ci[1]-mean]],fmt='D',capsize=6,color='#233d4f')
    axes[1].set_xticks([0,1],['Evolved − PPO','Random − PPO'])
    axes[1].set_ylabel('Paired return difference')
    axes[1].set_title('Exploratory 95% Student-t intervals across seeds')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.15)
    fig.suptitle('Structural learning-rule pilot · 999,424 training steps per agent')
    for ext in ('png','pdf'): fig.savefig(out/f'confirmation.{ext}',dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    parser.add_argument('--max-hours',type=float,default=2)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args(); out=args.out.resolve()
    prepare(out)
    if args.prepare: return
    gpus=[int(g) for g in args.gpus.split(',')]
    if not gpus or len(gpus)!=len(set(gpus)) or any(g<0 for g in gpus): raise ValueError('Invalid GPU pool')
    lock=(out/'controller.lock').open('a'); fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (out/'controller.log').open('a',buffering=1) as log:
        class Tee:
            def __init__(self,stream): self.stream=stream
            def write(self,value): self.stream.write(value); log.write(value)
            def flush(self): self.stream.flush(); log.flush()
        original_stdout=sys.stdout
        sys.stdout=Tee(original_stdout)
        try:
            validate_gpu_pipeline(out,gpus)
            if not args.validate_only:
                search=RuleSearch(out,gpus,args.max_hours); search.run()
        finally:
            sys.stdout=original_stdout


if __name__=='__main__': main()

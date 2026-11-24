"""Paired dual-temperature comparison for JSON-launched PPO continuations.

The shared PPO controllers train/evaluate each condition. This read-only
collector evaluates the original checkpoint on the same worlds and assembles
world-bootstrap comparisons as the controller evaluations finish.
"""
import argparse,csv,json,subprocess,time
from pathlib import Path
from run_reward_shaped_architectures import ROOT,PYTHON,environment,write_json
from cap_reward_shaped_runs import try_evaluation_lock


def read(path):return json.loads(Path(path).read_text())


def original_evaluation(root,protocol,temperature,deadline):
    name='original_t'+str(temperature).replace('.','p');out=root/name
    if (out/'summary.json').exists():return
    lock=None
    while lock is None:
        if time.time()>deadline:raise TimeoutError('Waiting for baseline evaluation GPU exceeded deadline')
        lock=try_evaluation_lock(protocol['evaluation_gpu'])
        if lock is None:time.sleep(5)
    try:
        command=[PYTHON,'-u',str(ROOT/'evaluate_reward_shaped_ppo.py'),
            '--checkpoint',protocol['source_checkpoint'],'--config-json',protocol['source_config'],
            '--world-seeds-file',protocol['evaluation_worlds'],'--repeats',str(protocol['repeats']),
            '--seed',str(protocol['evaluation_seed']),'--batch',str(protocol['batch']),
            '--temperature',str(temperature),'--out',str(out)]
        with (root/(name+'.log')).open('w') as log:
            subprocess.run(command,cwd=ROOT,env=environment(protocol['evaluation_gpu']),stdout=log,
                stderr=subprocess.STDOUT,check=True,timeout=max(1,deadline-time.time()))
    finally:lock.close()


def vector(folder):
    import numpy as np
    from collections import defaultdict
    grouped=defaultdict(list)
    with (folder/'episodes.csv').open() as f:
        for r in csv.DictReader(f):grouped[int(r['world_seed'])].append(int(r['all_22']=='True'))
    return tuple(sorted(grouped)),np.array([np.mean(grouped[k]) for k in sorted(grouped)])


def summarize(root,protocol):
    import numpy as np
    original={}
    entries=[]
    for t in (1.,.25):
        p=root/('original_t'+str(t).replace('.','p'))
        if (p/'summary.json').exists():
            original[t]=vector(p);entries.append(('original',p,t,1.,.01))
    for case in protocol['cases']:
        config=read(case['config'])
        for t,label in ((1.,'wallclock'),(.25,'wallclock_t0p25')):
            p=Path(case['study'])/'mlp'/f'evaluation_{label}'
            if (p/'summary.json').exists():
                entries.append((case['condition'],p,t,config['ppo']['policy_temperature'],config['ppo']['ent_coef']))
    if not entries:return
    rows=[]
    for condition,folder,t,train_t,entropy in entries:
        summary=read(folder/'summary.json');seeds,values=vector(folder)
        rng=np.random.default_rng(protocol['bootstrap_seed'])
        ix=rng.integers(0,len(values),(protocol['bootstrap_samples'],len(values)))
        ci=100*np.percentile(values[ix].mean(axis=1),[2.5,97.5])
        row=dict(condition=condition,training_temperature=train_t,entropy_coefficient=entropy,
            evaluation_temperature=t,episodes=summary['episodes'],all22_pct=100*values.mean(),
            all22_ci_low=ci[0],all22_ci_high=ci[1],
            diamond_pct=summary['achievement_success_pct']['collect_diamond'],
            plant_pct=summary['achievement_success_pct']['eat_plant'],
            mean_achievements=summary['mean_achievements'],mean_length=summary['mean_length'],
            additional_training_steps=0 if condition=='original' else summary['training_steps']-9999941632)
        for name,baseline_t in [('same_temperature',t),('original_t0p25',.25)]:
            if baseline_t not in original:continue
            old_seeds,baseline=original[baseline_t]
            if old_seeds!=seeds:raise ValueError('Mismatched evaluation worlds')
            delta=values-baseline;interval=100*np.percentile(delta[ix].mean(axis=1),[2.5,97.5])
            row['delta_vs_'+name+'_pp']=100*delta.mean()
            row['delta_vs_'+name+'_ci_low']=interval[0];row['delta_vs_'+name+'_ci_high']=interval[1]
        rows.append(row)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (root/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    lines=['# PPO temperature and entropy fine-tuning: three 50M-step continuations','',
        'Each run starts from the original 9,999,941,632-step feedforward PPO seed-50 checkpoint, trained with general death cost −2. All runs restore identical actor/critic weights, restart Adam, use LR 2e−5 annealed over 50M additional steps, 1,024 independent standard-reset environments, 64-step rollouts, and otherwise matching PPO settings. All train on ordinary random worlds; there is no failed-world reset bank. Each run has a nine-hour cap.','',
        '| Condition | Training temperature | Entropy coefficient | Training GPU |',
        '| --- | ---: | ---: | ---: |',
        '| Control | 1 | 0.01 | 2 |','| Lower temperature | 0.25 | 0.01 | 3 |',
        '| Lower entropy | 1 | 0.001 | 6 |','',
        'Training temperature consistently applies to action sampling, stored behavior likelihoods, new-policy likelihoods, and the entropy term. Temperature 1 returns the original distribution unchanged. Checkpoints are saved near +1M, +5M, +10M, +20M and at the actual final update.','',
        'Every final policy and the unchanged original are evaluated at BOTH T=1 and T=0.25 on 100 identical fresh original-generator worlds × three independent trials (300 episodes per cell). World draw seed 41550; action/dynamics evaluation seed 49550; 10,000-step cap. These worlds are disjoint from the earlier randomness sweep. No seed ID is exposed to the network. Success is single-episode all-22 completion, averaged across trials, not best-of-three. The original at T=0.25 is the benchmark to beat.','',
        'Evaluations share GPU 7 using an exclusive lock. Confidence intervals bootstrap worlds with 10,000 samples (seed 59650); paired differences retain each world’s three trials together. One training seed per condition: intervals do not measure training-seed variation.','',
        '| Condition | Evaluation T | All 22 (%) | Diamond (%) | Plant (%) |',
        '| --- | ---: | ---: | ---: | ---: |',
        *[f'| {r["condition"]} | {r["evaluation_temperature"]} | {r["all22_pct"]:.2f} | {r["diamond_pct"]:.2f} | {r["plant_pct"]:.2f} |' for r in rows],
        '', 'Full paired intervals are in `comparison.csv`. Each condition’s `mlp/run_manifest.json` contains its W&B URL; `mlp/progress.json` records training/evaluation progress. `comparison_progress.json` tracks this collector. All 22 achievement rates and per-episode outcomes remain in each evaluation folder.','']
    (root/'README.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    args=p.parse_args();root=args.study.resolve();protocol=read(root/'comparison_protocol.json')
    deadlines=[read(Path(c['study'])/'launch_manifest.json')['overall_deadline'] for c in protocol['cases']]
    deadline=min(deadlines)
    try:
        for t in (1.,.25):
            write_json(root/'comparison_progress.json',dict(state='evaluating_original',temperature=t,time=time.time()))
            original_evaluation(root,protocol,t,deadline);summarize(root,protocol)
        while time.time()<deadline:
            summarize(root,protocol)
            states={c['condition']:read(Path(c['study'])/'mlp/progress.json').get('state') for c in protocol['cases']}
            write_json(root/'comparison_progress.json',dict(state='waiting_for_continuations',training_states=states,time=time.time()))
            if all(s in ('complete','complete_with_failures','wall_clock_complete') for s in states.values()):
                write_json(root/'comparison_progress.json',dict(state='complete' if all(s=='complete' for s in states.values()) else 'complete_with_failures',training_states=states,time=time.time()))
                return
            time.sleep(10)
        raise TimeoutError('Comparison deadline reached')
    except Exception as e:
        write_json(root/'comparison_progress.json',dict(state='failed',error=str(e),time=time.time()));raise


if __name__=='__main__':main()

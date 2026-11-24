"""Frozen-policy temperature sweep with paired worlds and held-out confirmation.

JSON-configured evaluation only. Reuses evaluate_reward_shaped_ppo.py and never
constructs an optimizer or updates policy weights.
"""
import argparse,csv,hashlib,json,math,random,shutil,subprocess,time
from pathlib import Path
from run_reward_shaped_architectures import ROOT,PYTHON,environment,write_json,checkpoint_ready
from run_ppo_study import require_free
from cap_reward_shaped_runs import try_evaluation_lock


def read_json(path):return json.loads(Path(path).read_text())


def read_worlds(path,count):
    data=read_json(path);values=data['world_seeds'] if isinstance(data,dict) else data
    if (len(values)!=count or len(set(values))!=count or any(
            not isinstance(v,int) or isinstance(v,bool) or not 0<=v<2**32 for v in values)):
        raise ValueError('World bank must contain the requested number of distinct uint32 seeds')
    return values


def reserve_gpu(gpu,deadline):
    while time.time()<deadline:
        lock=try_evaluation_lock(gpu)
        if lock is not None:
            try:require_free(gpu);return lock
            except RuntimeError:lock.close()
        time.sleep(5)
    raise TimeoutError(f'GPU {gpu} stayed busy past the evaluation deadline')


def prepare(path,out):
    if (out/'launch_manifest.json').exists():raise FileExistsError(f'Already launched: {out}')
    spec=read_json(path)
    if not spec['temperatures'] or 1. not in spec['temperatures'] or len(set(spec['temperatures']))!=len(spec['temperatures']):
        raise ValueError('Use unique temperatures including 1.0')
    if any(not math.isfinite(t) or t<0 for t in spec['temperatures']):raise ValueError('Invalid temperature')
    if len(set(spec['gpus']))!=len(spec['gpus']) or not spec['gpus']:raise ValueError('Use distinct GPUs')
    if any(spec[k]<1 for k in ('worlds_per_split','repeats','batch','bootstrap_samples')):raise ValueError('Invalid count')
    checkpoint=(ROOT/spec['checkpoint']).resolve()
    if not checkpoint_ready(checkpoint):raise FileNotFoundError(f'Incomplete checkpoint: {checkpoint}')
    out.mkdir(parents=True,exist_ok=True)
    source=(ROOT/spec['source_config']).resolve()
    cfg=read_json(source)
    write_json(out/'inference_config.json',cfg)
    spec.update(checkpoint=str(checkpoint),source_config=str(source))
    seeds=random.Random(spec['world_generator_seed']).sample(range(2**32),2*spec['worlds_per_split'])
    banks={}
    for split,values in [('validation',seeds[:spec['worlds_per_split']]),('confirmation',seeds[spec['worlds_per_split']:])]:
        key=split+'_worlds_file'
        if spec.get(key):
            path=(ROOT/spec[key]).resolve();values=read_worlds(path,spec['worlds_per_split']);spec[key]=str(path)
        banks[split]=values
        bank=dict(world_seeds=values)
        if spec.get(key):bank['source_file']=spec[key]
        else:bank['generator_seed']=spec['world_generator_seed']
        write_json(out/f'{split}_worlds.json',bank)
    if set(banks['validation']).intersection(banks['confirmation']):
        raise ValueError('Validation and confirmation worlds must be disjoint')
    write_json(out/'experiment_config.json',spec)
    return spec


def label(t):return 'greedy' if t==0 else 'temperature_'+str(t).replace('.','p')


def phase(spec,out,split,temperatures,deadline):
    pending=list(temperatures);active={};completed=[]
    try:
        while pending or active:
            if time.time()>deadline:raise TimeoutError('Evaluation wall-clock cap reached')
            for gpu in spec['gpus']:
                if gpu in active or not pending:continue
                lock=try_evaluation_lock(gpu)
                if lock is None:continue
                try:require_free(gpu)
                except RuntimeError:lock.close();continue
                t=pending.pop(0);destination=out/split/label(t)
                destination.parent.mkdir(exist_ok=True)
                log=(destination.parent/(label(t)+'.log')).open('w')
                command=[PYTHON,'-u',str(ROOT/'evaluate_reward_shaped_ppo.py'),
                    '--checkpoint',spec['checkpoint'],'--config-json',str(out/'inference_config.json'),
                    '--world-seeds-file',str(out/f'{split}_worlds.json'),'--repeats',str(spec['repeats']),
                    '--batch',str(spec['batch']),'--seed',str(spec[f'{split}_rollout_seed']),
                    '--temperature',str(t),'--out',str(destination)]
                try:
                    process=subprocess.Popen(command,cwd=ROOT,env=environment(gpu),stdout=log,
                        stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                except BaseException:log.close();lock.close();raise
                active[gpu]=(process,log,t,lock)
            for gpu,(process,log,t,lock) in list(active.items()):
                code=process.poll()
                if code is None:continue
                log.close();lock.close();del active[gpu]
                if code:raise RuntimeError(f'{split} {t} failed ({code}); see its log')
                completed.append(t)
            write_json(out/'progress.json',dict(state='evaluating',split=split,
                completed_temperatures=completed,pending_temperatures=pending,
                active=[dict(gpu=g,pid=v[0].pid,temperature=v[2]) for g,v in active.items()],time=time.time()))
            if active or pending:time.sleep(5)
    finally:
        for process,log,t,lock in active.values():
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()
            log.close();lock.close()


def summarize(spec,out,split,temperatures):
    import numpy as np
    from collections import defaultdict
    vectors={};summaries={};joint_vectors={};raw_rows={}
    for t in temperatures:
        folder=out/split/label(t);summaries[t]=read_json(folder/'summary.json')
        grouped=defaultdict(list);joint=defaultdict(list)
        with (folder/'episodes.csv').open() as f:
            raw_rows[t]=list(csv.DictReader(f))
        for r in raw_rows[t]:
            grouped[int(r['world_seed'])].append(int(r['all_22']=='True'))
            joint[int(r['world_seed'])].append(int(r['all_22']=='True' and int(r['length'])==10000))
        seeds=sorted(grouped)
        if len(seeds)!=spec['worlds_per_split'] or any(len(v)!=spec['repeats'] for v in grouped.values()):
            raise ValueError('Missing world/repeat outcomes')
        vectors[t]=np.array([np.mean(grouped[s]) for s in seeds])
        joint_vectors[t]=np.array([np.mean(joint[s]) for s in seeds])
    rng=np.random.default_rng(spec['bootstrap_seed'])
    indices=rng.integers(0,spec['worlds_per_split'],(spec['bootstrap_samples'],spec['worlds_per_split']))
    rows=[]
    for t in temperatures:
        values=vectors[t];delta=values-vectors[1.]
        ci=100*np.percentile(values[indices].mean(axis=1),[2.5,97.5])
        paired_ci=100*np.percentile(delta[indices].mean(axis=1),[2.5,97.5])
        summary=summaries[t]
        trials=raw_rows[t];joint_ci=100*np.percentile(joint_vectors[t][indices].mean(axis=1),[2.5,97.5])
        rows.append(dict(temperature=t,action_selection=summary['action_selection'],episodes=summary['episodes'],
            all22_count=sum(r['all_22']=='True' for r in trials),
            hit_10k_count=sum(int(r['length'])==10000 for r in trials),
            timeout_count=sum(r['timeout']=='True' for r in trials),
            both_count=sum(r['all_22']=='True' and int(r['length'])==10000 for r in trials),
            both_pct=100*joint_vectors[t].mean(),both_ci_low=joint_ci[0],both_ci_high=joint_ci[1],
            all22_pct=100*values.mean(),all22_ci_low=ci[0],all22_ci_high=ci[1],
            paired_improvement_pp=100*delta.mean(),paired_ci_low=paired_ci[0],paired_ci_high=paired_ci[1],
            diamond_pct=summary['achievement_success_pct']['collect_diamond'],
            plant_pct=summary['achievement_success_pct']['eat_plant'],mean_achievements=summary['mean_achievements'],
            mean_length=summary['mean_length'],lava_pct=summary['lava_pct'],timeout_pct=summary['timeout_pct']))
    with (out/f'{split}_comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows


def write_report(spec,out,validation,confirmation):
    cfg=read_json(out/'inference_config.json')
    lines=['# Frozen-policy action randomness sweep','',
        f'Checkpoint: `{spec["checkpoint"]}`. Training seed {cfg.get("SEED")}; general death cost {cfg.get("DEATH_PENALTY",0)}; lava-only cost {cfg.get("LAVA_PENALTY",0)}. No network parameters were changed.','',
        f'Each setting uses {spec["worlds_per_split"]} fixed original-generator worlds × {spec["repeats"]} independent trials. Exact seed lists are saved in `validation_worlds.json` and `confirmation_worlds.json`; validation rollout seed {spec["validation_rollout_seed"]}; confirmation rollout seed {spec["confirmation_rollout_seed"]}. Within a split, all settings share world, action and environment random streams. Temperature scales policy logits; zero selects argmax. Dynamics remain stochastic even with greedy actions. Episode cap: 10,000.','',
        'Primary metric: single-trial completion of all 22, averaged across worlds/trials. This is not best-of-three success. Confidence intervals use 10,000 world-bootstrap samples; improvement intervals are paired against normal sampling. All 22 achievement rates and per-trial outcomes are retained in each evaluation folder.','',
        '## Validation sweep','',
        '| Temperature | All 22 (%) | Reach 10K | Both (%) | Diamond (%) | Plant (%) | Mean length |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
        *[f'| {"Greedy" if r["temperature"]==0 else r["temperature"]} | {r["all22_pct"]:.2f} | {r["hit_10k_count"]}/{r["episodes"]} | {r["both_pct"]:.2f} | {r["diamond_pct"]:.2f} | {r["plant_pct"]:.2f} | {r["mean_length"]:.0f} |' for r in validation]]
    if confirmation:
        lines+=['','## Independent confirmation','',
            'The validation winner and normal sampling were compared on a disjoint set of worlds. Validation scores used for selecting a setting are not unbiased estimates of its improvement.','',
            '| Temperature | All 22 (%) | Paired improvement (percentage points, 95% CI) |',
            '| --- | ---: | ---: |',
            *[f'| {"Greedy" if r["temperature"]==0 else r["temperature"]} | {r["all22_pct"]:.2f} | {r["paired_improvement_pp"]:+.2f} [{r["paired_ci_low"]:+.2f}, {r["paired_ci_high"]:+.2f}] |' for r in confirmation]]
    lines+=['','One frozen training seed. World-bootstrap intervals describe variation over these evaluation worlds, not variation across trained policies. Observed success on a finite set does not guarantee performance on all maps.','']
    (out/'README.md').write_text('\n'.join(lines))


def plot(out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold',
        'font.size':11,'legend.fontsize':9})
    x=np.arange(len(rows));y=np.array([r['all22_pct'] for r in rows])
    fig,ax=plt.subplots(figsize=(6.4,4.2))
    ax.errorbar(x,y,yerr=np.array([[r['all22_pct']-r['all22_ci_low'] for r in rows],
        [r['all22_ci_high']-r['all22_pct'] for r in rows]]),fmt='o-',color='#2563eb',capsize=4,label='All 22')
    ax.plot(x,[r['diamond_pct'] for r in rows],'s-',color='#dc2626',label='Diamond')
    ax.plot(x,[r['plant_pct'] for r in rows],'^-',color='#15803d',label='Plant')
    ax.set_xticks(x,['Normal\nT=1' if r['temperature']==1 else 'Greedy\nT=0' if r['temperature']==0 else f'T={r["temperature"]}' for r in rows])
    title=read_json(out/'experiment_config.json').get('display_name','Frozen-policy temperature sweep')
    ax.set(ylabel='Success rate (%)',xlabel='Action selection',ylim=(0,100),title=title)
    ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    for extension in ('pdf','png'):fig.savefig(out/f'randomness_sweep.{extension}',dpi=220)
    plt.close(fig)


def worker(out):
    spec=read_json(out/'experiment_config.json');deadline=read_json(out/'launch_manifest.json')['deadline']
    try:
        phase(spec,out,'validation',spec['temperatures'],deadline)
        validation=summarize(spec,out,'validation',spec['temperatures'])
        write_report(spec,out,validation,[]);plot(out,validation)
        best=max(validation,key=lambda row:(row['all22_pct'],row['both_pct'],row['temperature']==1.))['temperature']
        finalists=[1.] if best==1. else [1.,best]
        write_json(out/'selected_temperature.json',dict(temperature=best,selection_metric='validation all-22 success; ties use joint survival, then normal sampling'))
        phase(spec,out,'confirmation',finalists,deadline)
        confirmation=summarize(spec,out,'confirmation',finalists)
        write_report(spec,out,validation,confirmation)
        write_json(out/'progress.json',dict(state='complete',best_validation_temperature=best,time=time.time()))
    except Exception as error:
        write_json(out/'progress.json',dict(state='failed',error=str(error),time=time.time()));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--launch',action='store_true');p.add_argument('--worker',action='store_true')
    args=p.parse_args()
    if args.worker:worker(args.out.resolve());return
    if not args.config:raise ValueError('--config is required')
    name=read_json(args.config)['name'];out=(args.out or ROOT/'concept_mapping/runs'/name).resolve()
    spec=prepare(args.config,out)
    if not args.launch:print(f'Prepared only: {out}');return
    lock=reserve_gpu(spec['gpus'][0],time.time()+1800)
    try:
        with (out/'sampling_tests.log').open('w') as log:
            subprocess.run([PYTHON,'-m','unittest','-v','test_policy_temperature'],cwd=ROOT,
                env=environment(spec['gpus'][0]),stdout=log,stderr=subprocess.STDOUT,check=True)
    finally:lock.close()
    sources=['evaluate_reward_shaped_ppo.py','run_policy_randomness_sweep.py','test_policy_temperature.py']
    for source in sources:shutil.copy2(ROOT/source,out/source)
    start=time.time()
    write_json(out/'launch_manifest.json',dict(start=start,deadline=start+3600*spec['max_hours'],
        no_training=True,sources={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources}))
    with (out/'worker.log').open('w') as log:
        process=subprocess.Popen([PYTHON,'-u',str(Path(__file__).resolve()),'--worker','--out',str(out)],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    write_json(out/'worker.json',dict(pid=process.pid));print(f'Launched evaluator {process.pid}: {out}',flush=True)


if __name__=='__main__':main()

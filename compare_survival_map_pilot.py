"""Wait for paired studies, compare frozen outcomes, and confirm on fresh worlds."""
import argparse,csv,json,time,subprocess,re
from pathlib import Path
import numpy as np


def read_trials(path):
    with (path/'episodes.csv').open() as f:return list(csv.DictReader(f))


def compare(control,treatment):
    left=read_trials(control);right=read_trials(treatment)
    a=json.loads((control/'summary.json').read_text());b=json.loads((treatment/'summary.json').read_text())
    for key in ('evaluation_seed','episodes','temperature','max_episode_steps','training_steps'):
        if a[key]!=b[key]:raise ValueError(f'Unmatched comparison: {key}')
    if [r['episode'] for r in left]!=[r['episode'] for r in right]:raise ValueError('Episode IDs differ')
    metrics={'all_22':lambda r:r['all_22']=='True',
        'early_death':lambda r:int(r['length'])<1000 and (r['lava']=='True' or r['nonlava_death']=='True'),
        'timeout':lambda r:r['timeout']=='True'}
    rng=np.random.default_rng(7421);indices=rng.integers(0,len(left),(10000,len(left)));out={}
    for name,fn in metrics.items():
        x=np.array([fn(r) for r in left],float);y=np.array([fn(r) for r in right],float);delta=y-x
        out[name]=dict(control_pct=float(100*x.mean()),treatment_pct=float(100*y.mean()),
            treatment_minus_control_pp=float(100*delta.mean()),
            paired_bootstrap_ci95_pp=(100*np.quantile(delta[indices].mean(1),[.025,.975])).tolist())
    achievements=sorted(a['achievement_success_pct'])
    x=np.array([sum(int(r[k]) for k in achievements) for r in left],float)
    y=np.array([sum(int(r[k]) for k in achievements) for r in right],float)
    out['mean_achievements']=dict(control_mean=float(x.mean()),treatment_mean=float(y.mean()),
        difference=float((y-x).mean()),paired_bootstrap_ci95=np.quantile((y-x)[indices].mean(1),[.025,.975]).tolist())
    return dict(episodes=len(left),training_steps=a['training_steps'],evaluation_seed=a['evaluation_seed'],metrics=out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control',type=Path,required=True);p.add_argument('--treatment',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--watch',action='store_true')
    p.add_argument('--confirmation-seed',type=int,default=98150);p.add_argument('--confirmation-gpu',type=int,default=1)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    studies=[a.control.resolve()/'map_mlp_gru',a.treatment.resolve()/'map_mlp_gru']
    def status(**kw):(a.out/'progress.json').write_text(json.dumps(dict(time=time.time(),**kw),indent=2)+'\n')
    deadline=time.time()+144*3600
    def read_progress(study):
        try:return json.loads((study/'progress.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):return {}
    while True:
        progress=[read_progress(s) for s in studies]
        if all(d.get('state')=='complete' for d in progress):break
        if any(d.get('state') in ('complete_with_failures','wall_clock_complete') for d in progress):raise RuntimeError('A pilot did not finish cleanly; inspect its progress before comparing')
        if not a.watch:raise RuntimeError('Pilot is still running; use --watch for detached analysis')
        if time.time()>deadline:raise TimeoutError('Pilot comparison deadline reached')
        status(state='waiting_for_pilots',pilot_states=[d.get('state','not_started') for d in progress]);time.sleep(30)
    result=compare(*(s/'evaluation_wallclock' for s in studies))
    # Predeclared screening rule, not a statistical significance claim.
    d=result['metrics'];early=-d['early_death']['treatment_minus_control_pp'];joint=d['all_22']['treatment_minus_control_pp']
    promising=((early>=2 and joint>=-2) or (joint>=2 and early>=-2)) and d['mean_achievements']['difference']>=-.25
    result['promising_screen']=promising
    result['screen_rule']='At least 2 percentage points improvement in early deaths or all-22, with no more than 2 points worsening in the other metric and no more than 0.25 fewer mean achievements.'
    result['measured_training_sps']={}
    for label,study in zip(('control','treatment'),studies):
        matches=re.findall(r'^SPS:\s+([0-9.eE+-]+)',(study/'training.log').read_text(),re.MULTILINE)
        result['measured_training_sps'][label]=float(matches[-1]) if matches else None
    result['limitations']='One training seed; episode intervals do not quantify training-seed variation. Fresh confirmation uses fixed final checkpoints, not best-checkpoint selection.'
    (a.out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    if promising:
        from cap_reward_shaped_runs import try_evaluation_lock
        from run_reward_shaped_architectures import environment,PYTHON,ROOT
        for label,study in zip(('control','treatment'),studies):
            out=a.out/('fresh_'+label)
            if (out/'summary.json').exists():continue
            lock=None
            while lock is None:
                if time.time()>deadline:raise TimeoutError('Confirmation GPU never became available')
                lock=try_evaluation_lock(a.confirmation_gpu)
                if lock is None:status(state='waiting_for_confirmation_gpu',gpu=a.confirmation_gpu);time.sleep(30)
            try:
                status(state='confirming_on_fresh_maps',model=label)
                checkpoint=json.loads((study/'evaluation_wallclock/summary.json').read_text())['checkpoint']
                with (a.out/('fresh_'+label+'.log')).open('w') as log:
                    subprocess.run([PYTHON,'-u',str(ROOT/'evaluate_reward_shaped_ppo.py'),'--checkpoint',checkpoint,
                        '--config-json',str(study/'inference_config.json'),'--out',str(out),'--episodes','1024','--batch','32',
                        '--seed',str(a.confirmation_seed)],cwd=ROOT,env=environment(a.confirmation_gpu),stdout=log,stderr=subprocess.STDOUT,check=True)
            finally:lock.close()
        result['fresh_confirmation']=compare(a.out/'fresh_control',a.out/'fresh_treatment')
        (a.out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    status(state='complete',promising_screen=promising,fresh_confirmation_complete='fresh_confirmation' in result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

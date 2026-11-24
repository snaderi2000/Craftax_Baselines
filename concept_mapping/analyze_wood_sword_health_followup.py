"""Paired health-3/health-9 extension of the wood/no-sword zombie probe."""
import csv
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from concept_mapping.analyze_wood_sword_combat_followup import (
    BASE, save_csv, evaluate_values, _load_wandb_config, _init_network,
    _restore_train_state, make_craftax_env_from_name)

OUT = BASE / 'wood_vs_none_health3_vs9'


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--low-health',type=int,choices=[2,3],default=3)
    args=parser.parse_args()
    low=args.low_health
    OUT=BASE/f'wood_vs_none_health{low}_vs9'
    OUT.mkdir(exist_ok=True)
    source = list(csv.DictReader((BASE/'per_state_results.csv').open()))
    mask = np.array([r['original_strongest_sword'] in ('none','wood') for r in source])
    rows = [{k:v for k,v in r.items() if not k.startswith('V_') and '_vs_' not in k}
            for r,m in zip(source,mask) if m]
    with np.load(BASE/'state_bank.npz') as bank:
        assert all(int(r['seed'])==s and int(r['episode_id'])==e and int(r['timestep'])==t
                   for r,s,e,t in zip(source,bank['seed'],bank['episode_id'],bank['timestep']))
        factual=bank['observations'][mask];targets=bank['target_index'][mask]
    n=len(rows);assert n==4761
    run='wandb/run-20260630_214658-m0mw4end'
    config=_load_wandb_config(run)
    env=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False)
    net=_init_network(config,env,env.default_params)
    state,checkpoint=_restore_train_state(run,config,net,env,env.default_params,9999941632)
    fv=evaluate_values(state.params,net,factual,512)
    prior=np.array([float(r['V_factual']) for r,m in zip(source,mask) if m])
    assert np.allclose(fv,prior,atol=1e-5,rtol=1e-5)
    for i,r in enumerate(rows):r['V_factual']=float(fv[i])
    effects={}; edit_errors=[]
    for health in [low,9]:
        values={}
        for sword in ['wood','none']:
            present=factual.copy();present[:,1332:1335]=0;present[:,1335]=health/10
            if sword=='wood':present[:,1332]=.1
            assert np.all(np.rint(present[:,1332:1336]*10)==[int(sword=='wood'),0,0,health])
            keep=np.ones(1345,bool);keep[1332:1336]=False
            assert np.array_equal(present[:,keep],factual[:,keep])
            absent=present.copy();absent[np.arange(n),targets]=0
            assert np.all(present[np.arange(n),targets]==1)
            assert np.all(np.count_nonzero(present!=absent,axis=1)==1)
            for condition,obs in [('present',present),('absent',absent)]:
                key=sword+'_'+condition
                values[key]=evaluate_values(state.params,net,obs,512).astype(float)
                # Independent single-observation inference validates edited branches too.
                ref=np.array([float(np.asarray(net.apply(state.params,o[None])[1]).reshape(-1)[0]) for o in obs[:8]])
                assert np.allclose(ref,values[key][:8],atol=1e-5,rtol=1e-5)
                edit_errors.append(float(np.max(np.abs(ref-values[key][:8]))))
                for i,r in enumerate(rows):r[f'V_h{health}_{key}']=float(values[key][i])
        dp=values['wood_present']-values['none_present'];da=values['wood_absent']-values['none_absent']
        effects[str(health)]=np.column_stack([dp,da,dp-da])
    effects[f'{low}_minus_9']=effects[str(low)]-effects['9']
    summaries=[]
    for label,a in effects.items():
        for i,r in enumerate(rows):
            for j,name in enumerate(['present','absent','interaction']):r[f'h{label}_{name}']=float(a[i,j])
        for group in ['all','none','wood']:
            m=np.array([group=='all' or r['original_strongest_sword']==group for r in rows]);x=a[m]
            rng=np.random.default_rng(50);boot=[]
            for _ in range(100):boot.append(x[rng.integers(0,len(x),(100,len(x)))].mean(1))
            ci=np.percentile(np.concatenate(boot),[2.5,97.5],axis=0)
            r=dict(health_condition=label,group=group,n=len(x),median_interaction=float(np.median(x[:,2])),fraction_positive=float((x[:,2]>0).mean()))
            for j,name in enumerate(['present','absent','interaction']):
                r.update({f'mean_{name}':float(x[:,j].mean()),f'{name}_ci_low':float(ci[0,j]),f'{name}_ci_high':float(ci[1,j])})
            summaries.append(r)
    save_csv(OUT/'per_state_results.csv',rows);save_csv(OUT/'summary.csv',summaries)
    (OUT/'validation.json').write_text(json.dumps(dict(checkpoint=checkpoint,states=n,
        factual_reference_max_error=float(np.max(np.abs(fv-prior))),edited_single_call_max_error=max(edit_errors),
        checks='Exact sword counts and health; all other observation coordinates unchanged; only target zombie cleared.',
        bootstrap='10000 paired episode resamples, seed 50; paired health contrast uses same state in all eight branches.',
        policy='Frozen feed-forward; no training or rollouts.',
        caveat='Only observations edited. Hidden zombie health and resource counters not stored; achievements retained as metadata. Health setting awards no reward and advances no time.'),indent=2))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()

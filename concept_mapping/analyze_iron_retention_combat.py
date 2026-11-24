"""Iron-only removal x target presence at factual health, 3, and 9."""
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from concept_mapping.analyze_wood_sword_combat_followup import (
    BASE, save_csv, evaluate_values, _load_wandb_config, _init_network,
    _restore_train_state, make_craftax_env_from_name)

OUT=BASE/'iron_retention_health_followup'


def main():
    OUT.mkdir(exist_ok=True)
    source=list(csv.DictReader((BASE/'per_state_results.csv').open()))
    mask=np.array([int(r['inventory_iron_sword'])>0 for r in source])
    selected=[r for r,m in zip(source,mask) if m]
    rows=[{k:v for k,v in r.items() if not k.startswith('V_') and '_vs_' not in k} for r in selected]
    with np.load(BASE/'state_bank.npz') as bank:
        assert all(int(r['seed'])==s and int(r['episode_id'])==e and int(r['timestep'])==t for r,s,e,t in zip(source,bank['seed'],bank['episode_id'],bank['timestep']))
        factual=bank['observations'][mask];targets=bank['target_index'][mask]
    n=len(rows);assert n==2713
    assert len({(r['seed'],r['episode_id']) for r in rows})==n
    for r in rows:
        r['remaining_strongest_sword']='stone' if int(r['inventory_stone_sword']) else 'wood' if int(r['inventory_wood_sword']) else 'none'
    run='wandb/run-20260630_214658-m0mw4end';config=_load_wandb_config(run)
    env=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False)
    net=_init_network(config,env,env.default_params)
    state,checkpoint=_restore_train_state(run,config,net,env,env.default_params,9999941632)
    effects={};max_error=0.
    for label,health in [('regular',None),('health3',3),('health9',9)]:
        values={}
        base=factual.copy()
        if health is not None:base[:,1335]=health/10
        for iron in ['retained','removed']:
            present=base.copy()
            if iron=='removed':present[:,1334]=0
            expected=base.copy()
            if iron=='removed':expected[:,1334]=0
            assert np.array_equal(present,expected)
            allowed=np.ones(1345,bool);allowed[[1334,1335]]=False
            assert np.array_equal(present[:,allowed],factual[:,allowed])
            assert np.all(present[:,1334]>0) if iron=='retained' else np.all(present[:,1334]==0)
            absent=present.copy();absent[np.arange(n),targets]=0
            assert np.all(present[np.arange(n),targets]==1)
            assert np.all(np.count_nonzero(present!=absent,axis=1)==1)
            for condition,obs in [('present',present),('absent',absent)]:
                key=iron+'_'+condition
                values[key]=evaluate_values(state.params,net,obs,512).astype(float)
                for i,r in enumerate(rows):r[f'V_{label}_{key}']=float(values[key][i])
        if label=='regular':
            reference=np.array([float(r['V_factual']) for r in selected])
            max_error=float(np.max(np.abs(reference-values['retained_present'])))
            assert np.allclose(reference,values['retained_present'],atol=1e-5,rtol=1e-5)
        dp=values['retained_present']-values['removed_present']
        da=values['retained_absent']-values['removed_absent']
        effects[label]=np.column_stack([dp,da,dp-da])
    effects['health3_minus_health9']=effects['health3']-effects['health9']
    summaries=[]
    for label,a in effects.items():
        for i,r in enumerate(rows):
            for j,name in enumerate(['present','absent','interaction']):r[f'{label}_{name}']=float(a[i,j])
        for group in ['all','none','wood','stone']:
            m=np.array([group=='all' or r['remaining_strongest_sword']==group for r in rows]);x=a[m]
            result=dict(health_condition=label,remaining_sword=group,n=len(x))
            if len(x):
                rng=np.random.default_rng(50);boot=[]
                for _ in range(100):boot.append(x[rng.integers(0,len(x),(100,len(x)))].mean(1))
                ci=np.percentile(np.concatenate(boot),[2.5,97.5],axis=0)
                result.update(median_interaction=float(np.median(x[:,2])),fraction_positive=float((x[:,2]>0).mean()))
                for j,name in enumerate(['present','absent','interaction']):result.update({f'mean_{name}':float(x[:,j].mean()),f'{name}_ci_low':float(ci[0,j]),f'{name}_ci_high':float(ci[1,j]),f'{name}_fraction_positive':float((x[:,j]>0).mean())})
            summaries.append(result)
    # Include empty equipment strata explicitly, with blank estimates.
    keys=list(next(r for r in summaries if r['n']).keys())
    summaries=[{k:r.get(k,'') for k in keys} for r in summaries]
    save_csv(OUT/'per_state_results.csv',rows);save_csv(OUT/'summary.csv',summaries)
    (OUT/'validation.json').write_text(json.dumps(dict(checkpoint=checkpoint,n=n,factual_max_error=max_error,
        edits='Iron retained at factual count or set to zero. All wood/stone counts unchanged. Only target zombie channel cleared. Only health additionally changed in health3/health9.',
        remaining_group='Highest material tier remaining, not count-scaled damage.',
        bootstrap='10000 paired episode resamples per subgroup, seed 50, percentile 95% CI; one state per (seed, episode_id).',
        policy='Frozen feed-forward; no training, actions, reward awards, or rollouts.',
        missing='Latent counters and zombie health not stored. Full target health inferred from first attack. Achievement metadata unchanged.'),indent=2))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()

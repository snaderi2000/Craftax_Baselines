"""Wood versus no sword, crossed with immediate zombie presence; no rollouts."""
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from concept_mapping.analyze_stone_wood_pickaxe_interaction_final import (
    evaluate_values, _init_network, _load_wandb_config, _restore_train_state,
    make_craftax_env_from_name)

BASE = Path(__file__).resolve().parent / 'runs/combat_sword_interaction_10b_10000episodes'
OUT = BASE / 'wood_vs_none_followup'


def save_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    with (BASE/'per_state_results.csv').open() as f:
        source = list(csv.DictReader(f))
    mask = np.array([r['original_strongest_sword'] in ('none', 'wood') for r in source])
    selected = [r for r, keep in zip(source, mask) if keep]
    with np.load(BASE/'state_bank.npz') as bank:
        assert all(int(r['seed']) == s and int(r['episode_id']) == e and int(r['timestep']) == t
                   for r,s,e,t in zip(source,bank['seed'],bank['episode_id'],bank['timestep']))
        factual = bank['observations'][mask]; targets = bank['target_index'][mask]
    n = len(selected)
    assert n == 4761 and len({(r['seed'],r['episode_id']) for r in selected}) == n
    assert np.all(factual[np.arange(n),targets] == 1)
    assert np.all(factual[:,-1] == 0)
    assert np.all(factual[:,1333:1335] == 0)
    run = 'wandb/run-20260630_214658-m0mw4end'
    config = _load_wandb_config(run)
    env = make_craftax_env_from_name('Craftax-Classic-Symbolic-v1', auto_reset=False)
    net = _init_network(config, env, env.default_params)
    state, checkpoint = _restore_train_state(run, config, net, env, env.default_params,9999941632)
    factual_values = evaluate_values(state.params,net,factual,512)
    prior = np.array([float(r['V_factual']) for r in selected])
    assert np.allclose(factual_values,prior,atol=1e-5,rtol=1e-5)
    values = {}; validation = {}
    for sword in ['wood','none']:
        present = factual.copy(); present[:,1332:1335] = 0
        if sword == 'wood': present[:,1332] = .1
        absent = present.copy(); absent[np.arange(n),targets] = 0
        keep = np.ones(1345,bool); keep[1332:1335] = False
        assert np.array_equal(present[:,keep],factual[:,keep])
        assert np.all(present[:,1333:1335] == 0)
        assert np.all(np.rint(present[:,1332]*10) == (1 if sword=='wood' else 0))
        assert np.all(np.count_nonzero(present!=absent,axis=1)==1)
        assert np.all(absent[np.arange(n),targets]==0)
        for condition,obs in [('present',present),('absent',absent)]:
            key=sword+'_'+condition
            values[key]=evaluate_values(state.params,net,obs,512).astype(float)
            if sword=='wood':
                old=np.array([float(r['V_'+key]) for r in selected])
                validation[key+'_prior_max_error']=float(np.max(np.abs(values[key]-old)))
                assert np.allclose(values[key],old,atol=1e-5,rtol=1e-5)
    dp=values['wood_present']-values['none_present']
    da=values['wood_absent']-values['none_absent']
    interaction=dp-da
    rows=[]
    for i,r in enumerate(selected):
        row={k:v for k,v in r.items() if not k.startswith('V_') and '_vs_' not in k}
        row['V_factual']=float(factual_values[i])
        row.update({'V_'+k:float(v[i]) for k,v in values.items()})
        row.update(delta_present=float(dp[i]),delta_absent=float(da[i]),interaction=float(interaction[i]))
        rows.append(row)
    summaries=[]
    for group in ['all','none','wood']:
        m=np.array([group=='all' or r['original_strongest_sword']==group for r in selected])
        a=np.column_stack([dp[m],da[m],interaction[m]])
        rng=np.random.default_rng(50); samples=[]
        for _ in range(100):
            ix=rng.integers(0,len(a),(100,len(a)));samples.append(a[ix].mean(1))
        ci=np.percentile(np.concatenate(samples),[2.5,97.5],axis=0)
        row=dict(group=group,n=len(a),median_interaction=float(np.median(a[:,2])),fraction_positive=float((a[:,2]>0).mean()))
        for j,name in enumerate(['present','absent','interaction']):
            row.update({f'mean_{name}':float(a[:,j].mean()),f'{name}_ci_low':float(ci[0,j]),f'{name}_ci_high':float(ci[1,j])})
        summaries.append(row)
    save_csv(OUT/'per_state_results.csv',rows);save_csv(OUT/'summary.csv',summaries)
    validation.update(checkpoint=checkpoint,source=str(BASE),n=n,policy='feed-forward; no hidden state',
        factual_prior_max_error=float(np.max(np.abs(factual_values-prior))),
        edit_checks='Passed: sword counts exact, all other coordinates unchanged; target removal changes only target zombie channel.',
        selection='First successful attack, not first kill; original strongest wood or none.',
        bootstrap='10000 paired episode resamples within each group, seed 50, percentile 95% intervals.',
        achievements='Original flags retained as metadata; no awards or modifications. Not inputs to the symbolic critic.',
        limitations='Latent zombie health and resource counters are not saved. Observations alone are edited, with no actions, time advancement, training, or rollouts.')
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__': main()

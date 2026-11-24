"""Within-episode inventory swaps at 20%/80% of the saved action trajectory."""
import csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from concept_mapping.analyze_wood_sword_combat_followup import (
    save_csv,evaluate_values,_load_wandb_config,_init_network,_restore_train_state,make_craftax_env_from_name)

ROOT=Path(__file__).resolve().parent/'runs'
OUT=ROOT/'early_late_inventory_swap_10b_10000episodes'


def main():
    OUT.mkdir(exist_ok=True)
    early=[];late=[];rows=[]
    names=['wood','stone','coal','iron','diamond','sapling','wood_pickaxe','stone_pickaxe','iron_pickaxe','wood_sword','stone_sword','iron_sword']
    for seed,folder in [(50,'master_symbolic_10b_5000episodes'),(51,'master_symbolic_10b_5000episodes_seed51')]:
        for path in sorted((ROOT/folder/'episodes').glob('episode_*.npz')):
            with np.load(path) as d:
                n=len(d['actions']);te=int(np.floor(.2*(n-1)));tl=int(np.floor(.8*(n-1)))
                assert te<tl<n
                observations=d['observations']; e=observations[te].copy();l=observations[tl].copy()
                early.append(e);late.append(l)
                r=dict(seed=seed,episode_id=int(d['episode_id']),episode_steps=n,early_step=te,late_step=tl,
                    inventories_differ=bool(np.any(e[1323:1335]!=l[1323:1335])),
                    ended_in_lava=bool(d['dones'][-1] and d['terminal_observation'][:1323].reshape(7,9,21)[3,4,:17].argmax()==14),
                    collected_diamond=bool(d['terminal_achievements'][19]))
                inv=d['inventory'];intr=d['intrinsics'];ach=d['achievements']
                for prefix,t in [('early',te),('late',tl)]:
                    for k,v in zip(names,inv[t]):r[f'{prefix}_inventory_{k}']=int(v)
                    for k,v in zip(['health','food','water','energy'],intr[t]):r[f'{prefix}_{k}']=int(v)
                    r[prefix+'_achievement_count']=int(ach[t].sum())
                    r[prefix+'_achievement_flags']=json.dumps(ach[t].astype(int).tolist())
                rows.append(r)
            if len(rows)%1000==0:print('Extracted',len(rows),'pairs',flush=True)
    e=np.stack(early);l=np.stack(late);n=len(rows)
    assert n==10000 and len({(r['seed'],r['episode_id']) for r in rows})==n
    el=e.copy();el[:,1323:1335]=l[:,1323:1335]
    le=l.copy();le[:,1323:1335]=e[:,1323:1335]
    outside=np.ones(1345,bool);outside[1323:1335]=False
    assert np.array_equal(el[:,outside],e[:,outside]) and np.array_equal(le[:,outside],l[:,outside])
    assert np.array_equal(el[:,1323:1335],l[:,1323:1335]) and np.array_equal(le[:,1323:1335],e[:,1323:1335])
    np.savez_compressed(OUT/'paired_states.npz',early=e,late=l,seed=[r['seed'] for r in rows],episode_id=[r['episode_id'] for r in rows])
    run='wandb/run-20260630_214658-m0mw4end';config=_load_wandb_config(run)
    env=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False)
    net=_init_network(config,env,env.default_params)
    state,checkpoint=_restore_train_state(run,config,net,env,env.default_params,9999941632)
    values={};error=0.
    for key,obs in [('EE',e),('EL',el),('LE',le),('LL',l)]:
        values[key]=evaluate_values(state.params,net,obs,512).astype(float)
        ref=evaluate_values(state.params,net,obs[:32],1)
        error=max(error,float(np.max(np.abs(ref-values[key][:32]))))
        assert np.allclose(ref,values[key][:32],atol=1e-5,rtol=1e-5)
    de=values['EL']-values['EE'];dl=values['LL']-values['LE']
    for i,r in enumerate(rows):
        r.update({'V_'+k:float(v[i]) for k,v in values.items()})
        r.update(early_context_effect=float(de[i]),late_context_effect=float(dl[i]),
            average_inventory_effect=float((de[i]+dl[i])/2),interaction=float(dl[i]-de[i]),
            both_lower=bool(de[i]<0 and dl[i]<0),both_lower_tolerance_1e5=bool(de[i]<-1e-5 and dl[i]<-1e-5))
    save_csv(OUT/'per_episode_results.csv',rows)
    summaries=[]
    for group,mask in [('all',np.ones(n,bool)),('different_inventories',np.array([r['inventories_differ'] for r in rows])),('identical_inventories',np.array([not r['inventories_differ'] for r in rows]))]:
        a=de[mask];b=dl[mask]
        if not len(a):continue
        metrics=np.column_stack([a,b,(a+b)/2,(a<0),(b<0),(a<0)&(b<0),(a<-1e-5)&(b<-1e-5)])
        rng=np.random.default_rng(50);boot=[]
        for _ in range(100):boot.append(metrics[rng.integers(0,len(a),(100,len(a)))].mean(1))
        ci=np.percentile(np.concatenate(boot),[2.5,97.5],axis=0)
        s=dict(group=group,n=len(a),median_early_effect=float(np.median(a)),median_late_effect=float(np.median(b)),
            both_lower_count=int(((a<0)&(b<0)).sum()))
        for j,k in enumerate(['mean_early_effect','mean_late_effect','mean_inventory_effect','fraction_early_lower','fraction_late_lower','fraction_both_lower','fraction_both_lower_tolerance_1e5']):
            s[k]=float(metrics[:,j].mean());s[k+'_ci_low']=float(ci[0,j]);s[k+'_ci_high']=float(ci[1,j])
        summaries.append(s)
    save_csv(OUT/'summary.csv',summaries)
    validation=dict(checkpoint=checkpoint,n=n,selection='Deterministic floor(.2*(T-1)) and floor(.8*(T-1)), zero-based pre-action states; one pair per episode; not random timing.',
        inventory_slots='1323:1335, all twelve inventory entries; no health/resource/map/direction/light edits.',
        late_more_achievements=sum(r['late_achievement_count']>r['early_achievement_count'] for r in rows),
        identical_inventory_pairs=sum(not r['inventories_differ'] for r in rows),factual_and_edited_single_batch_max_error=error,
        bootstrap='10000 paired episode resamples within group, seed 50, percentile 95% intervals.',
        policy='Frozen feed-forward; no training or rollouts; all observation-edit assertions passed.',
        interpretation='Tests whether later inventory lowers value in both contexts, not dominance over other signals or why it does so. Timing is relative to observed episode length, not an achievement-stage definition.')
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2))
    print(json.dumps(summaries,indent=2),flush=True)
    print(json.dumps(validation,indent=2),flush=True)


if __name__=='__main__':main()

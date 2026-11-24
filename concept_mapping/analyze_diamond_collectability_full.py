"""All eligible first-diamond states: tool+diamond highest-value criterion."""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from concept_mapping.analyze_wood_sword_combat_followup import (
    save_csv,evaluate_values,_load_wandb_config,_init_network,_restore_train_state,make_craftax_env_from_name)
ROOT=Path(__file__).resolve().parent/'runs'
OUT=ROOT/'diamond_collectability_10b_10000episodes_all_visible_stone'


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--resource',choices=['diamond','iron'],default='diamond')
    resource=parser.parse_args().resource
    achievement,block,tool,materials,expected = ((19,10,8,[0,1,2,3],(3483,21,6496)) if resource=='diamond' else (18,9,7,[0,1],(6567,3281,152)))
    slot=1323+tool
    tool_name='iron pickaxe' if resource=='diamond' else 'stone pickaxe'
    OUT=ROOT/f'{resource}_collectability_10b_10000episodes_all_visible_stone'
    OUT.mkdir(exist_ok=True);rows=[];observations=[];excluded=0;no_event=0
    for seed,folder in [(50,'master_symbolic_10b_5000episodes'),(51,'master_symbolic_10b_5000episodes_seed51')]:
        for path in sorted((ROOT/folder/'episodes').glob('episode_*.npz')):
            with np.load(path) as d:
                ach=d['achievements'];ts=np.flatnonzero(np.diff(np.r_[ach[:,achievement],d['terminal_achievements'][achievement]].astype(int))>0)
                if not len(ts):no_event+=1;continue
                t=int(ts[0]);inv=d['inventory'][t]
                if all(inv[k]>=1 for k in materials):excluded+=1;continue
                direction=int(d['direction'][t]);rr=3+[0,0,-1,1][direction];cc=4+[-1,1,0,0][direction]
                assert d['actions'][t]==5 and not d['sleeping'][t] and inv[tool]>0
                assert d['visible_blocks'][t,rr,cc]==block and not ach[t,achievement]
                o=d['observations'][t].copy();observations.append(o)
                rows.append(dict(seed=seed,episode_id=int(d['episode_id']),timestep=t,target_row=rr,target_col=cc,
                    inventory=json.dumps(inv.astype(int).tolist()),achievement_flags=json.dumps(ach[t].astype(int).tolist()),
                    health=int(d['intrinsics'][t,0]),food=int(d['intrinsics'][t,1]),water=int(d['intrinsics'][t,2]),energy=int(d['intrinsics'][t,3])))
    factual=np.stack(observations);n=len(rows);assert (n,excluded,no_event)==expected
    assert len({(r['seed'],r['episode_id']) for r in rows})==n
    np.savez_compressed(OUT/'base_states.npz',observations=factual,seed=[r['seed'] for r in rows],episode_id=[r['episode_id'] for r in rows])
    print('Validated bank:',n,flush=True)
    run='wandb/run-20260630_214658-m0mw4end';cfg=_load_wandb_config(run)
    env=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False)
    net=_init_network(cfg,env,env.default_params)
    state,checkpoint=_restore_train_state(run,cfg,net,env,env.default_params,9999941632)
    ore=factual[:,:1323].reshape(n,63,21)[:,:,block]==1
    values={}
    for key,tool,present in [('A',True,True),('B',False,True),('C',False,False),('D',True,False)]:
        obs=factual.copy()
        if not tool:obs[:,slot]=0
        if not present:
            view=obs[:,:1323].reshape(n,63,21)
            view[:,:,block][ore]=0;view[:,:,4][ore]=1
            assert not np.any(view[:,:,block])
            assert np.all(np.sum(obs[:,:1323]!=factual[:,:1323],axis=1)==2*ore.sum(1))
        else:assert np.array_equal(obs[:,:1323],factual[:,:1323])
        assert np.array_equal(obs[:,slot+1:],factual[:,slot+1:])
        assert np.array_equal(obs[:,1323:slot],factual[:,1323:slot])
        if not tool:assert np.all(obs[:,slot]==0)
        values[key]=evaluate_values(state.params,net,obs,512).astype(float)
    a,b,c,d=[values[k] for k in 'ABCD']
    margin=a-np.maximum.reduce([b,c,d]);success=margin>0
    strict=success&(np.minimum(b,c)>d)
    metrics=np.column_stack([success,a>b,a>c,a>d,strict,margin,margin>1e-5])
    rng=np.random.default_rng(50);boot=[]
    for _ in range(100):boot.append(metrics[rng.integers(0,n,(100,n))].mean(1))
    ci=np.percentile(np.concatenate(boot),[2.5,97.5],axis=0)
    summary=dict(n=n,resource=resource,excluded_recraft=excluded,no_collection_episodes=no_event,checkpoint=checkpoint,success_count=int(success.sum()))
    for j,k in enumerate(['success_rate','A_gt_B','A_gt_C','A_gt_D','previous_strict_ordering_rate','mean_margin','success_rate_margin_gt_1e5']):
        summary[k]=float(metrics[:,j].mean());summary[k+'_95ci']=ci[:,j].tolist()
    for i,r in enumerate(rows):
        r.update({'V_'+k:float(v[i]) for k,v in values.items()})
        r.update(visible_resource_tiles=int(ore[i].sum()),success=bool(success[i]),margin=float(margin[i]),previous_strict_success=bool(strict[i]))
    # Cross-check previously evaluated pilot examples, including all four branches.
    pilot=list(csv.DictReader((ROOT/'collectability_three_tasks_10b_pilot'/resource/'per_state_results.csv').open()))
    lookup={(r['seed'],r['episode_id'],r['timestep']):r for r in rows};err=0.
    for p in pilot:
        r=lookup[tuple(int(p[k]) for k in ['seed','episode_id','timestep'])]
        for k in 'ABCD':
            err=max(err,abs(r['V_'+k]-float(p['V_'+k])))
            assert np.isclose(r['V_'+k],float(p['V_'+k]),atol=1e-5,rtol=1e-5)
    summary.update(pilot_reference_max_error=err,bootstrap='10000 paired episode resamples, seed 50; percentile 95% intervals.',
        criterion=f'A > max(B,C,D); A={resource}+pickaxe, B={resource} without pickaxe, C=no {resource}/no pickaxe, D=no {resource}+pickaxe.',
        edits=f'Retain factual {tool_name} count or remove only {tool_name}. Replace ALL visible {resource} tiles with stone; all other observation features unchanged. No actions/rewards/training/rollouts.',
        selection=f'Immediately before first successful {resource} collection per episode; exclude sufficient inventory to recraft {tool_name}, regardless of station access.',recraft_inventory_indices=materials)
    save_csv(OUT/'per_state_results.csv',rows);(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

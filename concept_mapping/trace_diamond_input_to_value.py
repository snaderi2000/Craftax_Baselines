"""Trace input-to-value diamond circuits and predict held-out interventions.

The original PPO remains frozen. Graphs consist of real tanh neurons/weights;
removed incoming messages are replaced by discovery-set reference means. Graph
size is selected on discovery episodes only. Evaluation includes original held-
out states and fresh episode-disjoint, natural diamond-visible states.
"""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('concept_mapping/runs/diamond_critic_circuit_10b')
OUT=ROOT/'input_to_value_circuit'
RUN='wandb/run-20260630_214658-m0mw4end'
STEP=9999941632
TARGET=np.array([147,198,11,439])
Q=np.array([1.,-1.,1.,-1.])
ORDER=[0,1,3,2]
LABELS=['Both','Diamond only','Pickaxe only','Neither']
SIZES=[1,2,4,8,16,32,64,128,512]


def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))


def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def fresh_sample(out):
    if (out/'fresh_base_states.npz').exists():
        with np.load(out/'fresh_base_states.npz') as d:obs=d['observations']
        return obs,read(out/'fresh_state_selection.csv')
    old={(int(r['seed']),int(r['episode_id'])) for r in read(ROOT/'episode_split.csv')}
    prior={(int(r['seed']),int(r['episode_id'])) for r in read(ROOT/'random_natural_pickaxe_probe/sampled_states.csv')}
    excluded=old|prior;manifest=[]
    for seed,name in [(50,'master_symbolic_10b_5000episodes'),(51,'master_symbolic_10b_5000episodes_seed51')]:
        for path in sorted((Path('concept_mapping/runs')/name/'episodes').glob('episode_*.npz')):
            eid=int(path.stem.split('_')[-1])
            if (seed,eid) not in excluded:manifest.append((seed,eid,path))
    rng=np.random.default_rng(150198);obs=[];rows=[]
    for seed,eid,path in manifest:
        with np.load(path) as d:
            blocks=d['visible_blocks'];inv=d['inventory']
            valid=np.flatnonzero((blocks==10).any((1,2))&(inv[:,8]>0)&(d['sleeping']==0))
            if not len(valid):continue
            t=int(rng.choice(valid));coords=np.argwhere(blocks[t]==10)
            direction=int(d['direction'][t]);dr,dc=[(0,-1),(0,1),(-1,0),(1,0)][direction]
            rows.append(dict(seed=seed,episode_id=eid,timestep=t,distance_to_diamond=int(np.abs(coords-[3,4]).sum(1).min()),
                diamond_in_front=int(blocks[t,3+dr,4+dc]==10),visible_diamonds=len(coords),
                diamond_already_collected=int(d['achievements'][t,19]),achievement_count=int(d['achievements'][t].sum()),
                health=int(d['intrinsics'][t,0]),source_episode=str(path)))
            obs.append(d['observations'][t].astype(np.float32))
    write(out/'fresh_state_selection.csv',rows)
    np.savez_compressed(out/'fresh_base_states.npz',observations=np.stack(obs))
    (out/'fresh_sampling.json').write_text(json.dumps(dict(seed=150198,original_bank_episodes_excluded=len(old),
        previous_random_probe_episodes_excluded=len(prior),total_excluded=len(excluded),episodes_scanned=len(manifest),
        eligible_episodes=len(rows),selection='One uniformly random awake diamond-visible iron-pickaxe-equipped timestep per eligible episode; all eligible episodes used. No value, activation, or future success filter.'),indent=2)+'\n')
    return np.stack(obs),rows


def branches(factual):
    a=factual.copy();b=a.copy();b[:,1331]=0
    d=a.copy();view=d[:,:1323].reshape(-1,63,21);mask=view[:,:,10]==1
    view[:,:,10][mask]=0;view[:,:,4][mask]=1
    c=d.copy();c[:,1331]=0
    assert np.all(a[:,1331]>0) and np.all(mask.sum(1)>0)
    assert np.all(np.count_nonzero(a!=b,axis=1)==1)
    assert np.all(np.count_nonzero(a!=d,axis=1)==2*mask.sum(1))
    return np.stack([a,b,c,d])


def load_weights(out):
    path=out/'frozen_critic_weights.npz'
    if not path.exists():
        from concept_mapping.plot_collectability_emergence import _load_wandb_config,_init_network,_restore_train_state,make_craftax_env_from_name
        cfg=_load_wandb_config(RUN);env=make_craftax_env_from_name(cfg['ENV_NAME'],auto_reset=False)
        net=_init_network(cfg,env,env.default_params)
        state,_=_restore_train_state(RUN,cfg,net,env,env.default_params,STEP)
        p=state.params['params'];data={}
        for j,i in enumerate(range(4,8)):
            data[f'W{j}']=np.asarray(p[f'Dense_{i}']['kernel'])
            data[f'b{j}']=np.asarray(p[f'Dense_{i}']['bias'])
        np.savez_compressed(path,**data)
    with np.load(path) as d:
        return [d[f'W{j}'] for j in range(4)],[d[f'b{j}'] for j in range(4)]


def forward(x,W,b):
    h=[x];z=[]
    for j in range(3):
        zz=h[-1]@W[j]+b[j];z.append(zz);h.append(np.tanh(zz))
    return h,z,(h[-1]@W[3]+b[3]).squeeze(-1)


def secant(za,zb,ha,hb):
    dz=za-zb
    return np.divide(ha-hb,dz,out=(1-((ha+hb)*.5)**2),where=np.abs(dz)>1e-5)


def rank_edges(x,W,b,discovery,out):
    xd=x[:,discovery];h,z,v=forward(xd,W,b)
    ref=[a.mean((0,1)) for a in h]
    w=np.zeros(512,dtype=np.float32);w[TARGET]=W[3].reshape(-1)[TARGET]
    scores=[np.zeros_like(W[j],dtype=np.float64) for j in range(3)]
    # Exact finite-difference chain factors are discovery ranking heuristics.
    # Their use does not establish that a pruned graph is causally faithful.
    max_error=0.
    def accumulate(ha,hb,za,zb,weight):
        nonlocal max_error
        gains=[secant(za[j],zb[j],ha[j+1],hb[j+1]) for j in range(3)]
        g=[None,None,gains[2]*w]
        g[1]=(g[2]@W[2].T)*gains[1]
        g[0]=(g[1]@W[1].T)*gains[0]
        goal=(ha[3]-hb[3])@w
        for j in range(3):
            delta=ha[j]-hb[j]
            contribution=np.abs(delta).T@np.abs(g[j])
            scores[j][:]+=weight*contribution*np.abs(W[j])/len(delta)
            replay=((delta@W[j])*g[j]).sum(1)
            max_error=max(max_error,float(np.max(np.abs(replay-goal))))
    for a,c in [(0,3),(1,2),(0,1),(3,2)]:
        accumulate([a_[a] for a_ in h],[a_[c] for a_ in h],[a_[a] for a_ in z],[a_[c] for a_ in z],.25)
    # Background ranking includes actual context variation relative to a fixed
    # mean observation, using the same exact chain-factor calculation.
    hm,zm,_=forward(xd.mean((0,1))[None],W,b)
    for c in range(4):
        accumulate([a_[c] for a_ in h],hm,[a_[c] for a_ in z],zm,.25)
    rankings=[np.argsort(-s,axis=0,kind='stable') for s in scores]
    np.savez_compressed(out/'discovery_rankings.npz',**{f'score{j}':s for j,s in enumerate(scores)},
        **{f'ranking{j}':r for j,r in enumerate(rankings)},**{f'reference{j}':r for j,r in enumerate(ref)})
    print(f'Discovery-only rankings complete; chain replay max error {max_error:.3g}.',flush=True)
    return rankings,ref,max_error


def build_graph(k,rankings,ref,W,b,rng=None):
    masks=[np.zeros_like(W[j],dtype=np.float32) for j in range(3)]
    children=TARGET.copy();sizes=[]
    for j in [2,1,0]:
        take=min(4*k if j==0 else k,len(W[j]))
        for child in children:
            selected=rankings[j][:take,child] if rng is None else rng.choice(len(W[j]),take,replace=False)
            masks[j][selected,child]=1
        children=np.flatnonzero(masks[j].any(axis=1))
        sizes.append(len(children))
    ww=[W[j]*masks[j] for j in range(3)]
    bb=[b[j]+ref[j]@(W[j]*(1-masks[j])) for j in range(3)]
    wo=np.zeros_like(W[3]);wo[TARGET]=W[3][TARGET]
    bo=b[3]+ref[3]@(W[3]-wo)
    return dict(k=k,masks=masks,W=ww+[wo],b=bb+[bo],input_nodes=sizes[2],l1_nodes=sizes[1],l2_nodes=sizes[0],
        edges=int(sum(m.sum() for m in masks)+len(TARGET)),nodes=int(sum(sizes)+len(TARGET)+1))


def metrics(name,k,h,v,fh,fv,W,mask=None):
    if mask is None:mask=np.ones(v.shape[1],bool)
    original=(Q@fv)[mask];predicted=(Q@v)[mask]
    h3=h[3][:,mask][:,:,TARGET];truth=fh[3][:,mask][:,:,TARGET]
    scale=max(float(np.abs(original).mean()),1e-8)
    return dict(population=name,k=k,n=int(mask.sum()),original_interaction_mean=float(original.mean()),
        predicted_interaction_mean=float(predicted.mean()),interaction_retained_pct=float(100*predicted.mean()/original.mean()) if abs(original.mean())>1e-8 else None,
        interaction_mae=float(np.abs(predicted-original).mean()),interaction_normalized_mae=float(np.abs(predicted-original).mean()/scale),
        interaction_sign_agreement_pct=float(100*((predicted>0)==(original>0)).mean()),
        full_value_mae=float(np.abs(v[:,mask]-fv[:,mask]).mean()),
        target_activation_mae=float(np.abs(h3-truth).mean()),
        isolated_target_value_mae=float(np.abs((h3-truth)@W[3][TARGET]).mean()),
        h198_mae=float(np.abs(h[3][:,mask,198]-fh[3][:,mask,198]).mean()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--bank-dir',type=Path,default=Path('concept_mapping/runs/diamond_collectability_10b_10000episodes_all_visible_stone'))
    parser.add_argument('--external-dir',type=Path,default=OUT)
    parser.add_argument('--random-draws',type=int,default=20)
    parser.add_argument('--seed',type=int,default=50)
    parser.add_argument('--bootstrap-draws',type=int,default=10000)
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True);began=time.monotonic()
    with np.load(args.bank_dir/'base_states.npz') as d:
        old=branches(d['observations'])
    split=read(args.bank_dir/'episode_split.csv' if (args.bank_dir/'episode_split.csv').exists() else ROOT/'episode_split.csv')
    discovery=np.array([int(r['state_index']) for r in split if r['split']=='discovery'])
    heldout=np.array([int(r['state_index']) for r in split if r['split']=='heldout'])
    if out!=args.external_dir:
        for filename in ['fresh_base_states.npz','fresh_state_selection.csv','fresh_sampling.json','frozen_critic_weights.npz']:
            source=args.external_dir/filename
            if source.exists() and not (out/filename).exists():shutil.copy2(source,out/filename)
    fresh,fr=fresh_sample(out);xf=branches(fresh)
    old_ids={(int(r['seed']),int(r['episode_id'])) for r in split}
    prior_ids={(int(r['seed']),int(r['episode_id'])) for r in read(ROOT/'random_natural_pickaxe_probe/sampled_states.csv')}
    fresh_ids={(int(r['seed']),int(r['episode_id'])) for r in fr}
    assert len(fr)==len(fresh_ids) and not fresh_ids&(old_ids|prior_ids)
    W,b=load_weights(out);weight_hash=digest(out/'frozen_critic_weights.npz')
    protocol=dict(run=RUN,checkpoint=STEP,bank_dir=str(args.bank_dir),targets=TARGET.tolist(),discovery_n=len(discovery),heldout_n=len(heldout),fresh_n=len(fr),
        discovery_episodes=len({(split[i]['seed'],split[i]['episode_id']) for i in discovery}),
        heldout_episodes=len({(split[i]['seed'],split[i]['episode_id']) for i in heldout}),
        hypothesis='A sparse network of original weighted tanh connections predicts conditional diamond/tool valuation and unseen internal intervention effects.',
        ranking='Discovery mean absolute finite-difference path contributions, equally weighted across four diamond/pickaxe contrasts, plus equally weighted background contrasts against a fixed mean observation. Ranking only; causal fidelity tested by actual nonlinear execution.',
        sizes=SIZES,input_parents_per_node='min(4*k,1345)',hidden_parents_per_node='k',
        selection='Smallest discovery-only k with normalized interaction MAE<=0.15, mean interaction retention between 90% and 110%, and four-target activation MAE<=0.15. If none qualifies, k=128 is reported as a failed compactness diagnostic.',
        omitted_messages='Fixed pooled discovery means, identical across conditions and evaluation episodes. Non-target final-hidden value contribution fixed to discovery mean.',
        fresh_selection=json.loads((out/'fresh_sampling.json').read_text()),
        testing='Original held-out bank plus fresh episodes excluded from both original diamond bank and prior random-state probe; fresh natural diamond-visible states need not be facing diamond.',
        interpretation='All-visible diamond-to-stone replacement and iron-pickaxe removal; no actions, time changes, training or new rollouts.',
        random_graph_draws=args.random_draws,bootstrap_draws=args.bootstrap_draws,seed=args.seed)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    fh,fz,fv=forward(old,W,b);fresh_h,fresh_z,fresh_v=forward(xf,W,b)
    baseline=read(args.bank_dir/'per_state_results.csv')
    expected=np.array([[float(r['V_'+key]) for r in baseline] for key in 'ABCD'])
    err=float(np.max(np.abs(fv-expected)));np.testing.assert_allclose(fv,expected,atol=1e-5,rtol=1e-5)
    print(f'Full critic replay max error: {err:.3g}; fresh episodes: {len(fr)}.',flush=True)
    rankings,ref,chain_error=rank_edges(old,W,b,discovery,out)
    selection=[];graphs={}
    for k in SIZES:
        graph=build_graph(k,rankings,ref,W,b);graphs[k]=graph
        hh,_,vv=forward(old[:,discovery],graph['W'],graph['b'])
        m=metrics('discovery',k,hh,vv,[a[:,discovery] for a in fh],fv[:,discovery],W)
        m.update({key:graph[key] for key in ['edges','nodes','input_nodes','l1_nodes','l2_nodes']});selection.append(m)
        print(f'Discovery k={k}: {graph["edges"]} edges, normalized interaction MAE={m["interaction_normalized_mae"]:.3f}, retention={m["interaction_retained_pct"]:.1f}%, activation MAE={m["target_activation_mae"]:.3f}',flush=True)
    passing=[m for m in selection if m['interaction_normalized_mae']<=.15 and 90<=m['interaction_retained_pct']<=110 and m['target_activation_mae']<=.15]
    chosen=passing[0]['k'] if passing else 128;graph=graphs[chosen]
    decision=dict(chosen_k=chosen,discovery_criteria_met=bool(passing),chosen_discovery_metrics=next(m for m in selection if m['k']==chosen),
        fresh_evaluation_not_used_for_selection=True)
    (out/'selection.json').write_text(json.dumps(decision,indent=2)+'\n');write(out/'discovery_selection.csv',selection)
    print('Frozen selection: '+json.dumps(decision),flush=True)
    results=[];outputs={};populations={'heldout':(old[:,heldout],[a[:,heldout] for a in fh],fv[:,heldout]),'fresh':(xf,fresh_h,fresh_v)}
    for name,(xx,truth,values) in populations.items():
        for k in SIZES:
            hh,zz,vv=forward(xx,graphs[k]['W'],graphs[k]['b'])
            m=metrics(name,k,hh,vv,truth,values,W);m.update({key:graphs[k][key] for key in ['edges','nodes']});results.append(m)
            if k==chosen:outputs[name]=(hh,zz,vv)
        print(f'{name} sparse graphs evaluated.',flush=True)
    write(out/'circuit_fidelity.csv',results)
    # Fresh-state strata are descriptive and never affect candidate selection.
    strata=[]
    for name,mask in [('all',np.ones(len(fr),bool)),('diamond_in_front',np.array([int(r['diamond_in_front'])>0 for r in fr])),
        ('not_in_front',np.array([int(r['diamond_in_front'])==0 for r in fr])),
        ('distance_1_2',np.array([int(r['distance_to_diamond'])<=2 for r in fr])),
        ('distance_3_7',np.array([int(r['distance_to_diamond'])>2 for r in fr]))]:
        if mask.any():strata.append(metrics(name,chosen,outputs['fresh'][0],outputs['fresh'][2],fresh_h,fresh_v,W,mask))
    write(out/'fresh_strata.csv',strata)
    heldout_strata=[]
    heldout_groups=np.array([split[int(i)].get('stratum','immediate_collection') for i in heldout])
    for label in sorted(set(heldout_groups)):
        heldout_strata.append(metrics(label,chosen,outputs['heldout'][0],outputs['heldout'][2],
            [a[:,heldout] for a in fh],fv[:,heldout],W,heldout_groups==label))
    write(out/'heldout_strata.csv',heldout_strata)
    coverage,diagnostics=input_diagnostics(out,graph,ref,W,b,old,xf,fresh_h,fresh_v)
    # Endpoint: with every incoming edge restored, four-target responses and
    # their value contrast must reproduce the true model exactly.
    all_h,_,all_v=forward(xf,graphs[512]['W'],graphs[512]['b'])
    np.testing.assert_allclose(all_h[3][:,:,TARGET],fresh_h[3][:,:,TARGET],atol=1e-5,rtol=1e-5)
    target_w=W[3].reshape(-1)[TARGET]
    np.testing.assert_allclose(Q@all_v,np.einsum('c,cni,i->n',Q,fresh_h[3][:,:,TARGET],target_w),atol=1e-5)
    # Save fixed graph edges, discovery-only scores and omitted-message biases.
    edge_rows=[]
    with np.load(out/'discovery_rankings.npz') as saved_scores:
        scores=[saved_scores[f'score{j}'] for j in range(3)]
    for j,mask in enumerate(graph['masks']):
        for source,target in np.argwhere(mask):
            edge_rows.append(dict(layer=j+1,source=int(source),target=int(target),weight=float(W[j][source,target]),
                discovery_score=float(scores[j][source,target])))
    write(out/'selected_edges.csv',edge_rows)
    np.savez_compressed(out/'selected_graph.npz',**{f'W{j}':w for j,w in enumerate(graph['W'])},
        **{f'b{j}':bb for j,bb in enumerate(graph['b'])},**{f'mask{j}':m for j,m in enumerate(graph['masks'])})
    records=[]
    for population,(xx,truth,values) in populations.items():
        hh,_,vv=outputs[population]
        ids=[split[int(i)] for i in heldout] if population=='heldout' else fr
        for i,row in enumerate(ids):
            record=dict(population=population,seed=row['seed'],episode_id=row['episode_id'],timestep=row['timestep'],stratum=row.get('stratum','external_random'),
                original_interaction=float((Q@values)[i]),predicted_interaction=float((Q@vv)[i]))
            for c,key in enumerate('ABCD'):
                record[f'V_{key}']=float(values[c,i]);record[f'V_{key}_circuit']=float(vv[c,i])
                for target in TARGET:
                    record[f'h{target}_{key}']=float(truth[3][c,i,target]);record[f'h{target}_{key}_circuit']=float(hh[3][c,i,target])
            records.append(record)
    write(out/'per_state_predictions.csv',records)
    # Independent model effects: no gain recalculation or fitting after test
    # perturbations. Candidate forward passes predict actual downstream effects.
    intervention_rows,intervention_detail=interventions(out,rankings,graph,W,b,populations,outputs)
    write(out/'intervention_predictions.csv',intervention_rows);write(out/'per_state_interventions.csv',intervention_detail)
    # Background messages: quantify broad natural variation entering neuron 198.
    background=background_analysis(out,rankings,ref,W,b,old[:,discovery],populations)
    random=[];rng=np.random.default_rng(args.seed+159)
    for draw in range(args.random_draws):
        random_graph=build_graph(chosen,rankings,ref,W,b,rng)
        hh,_,vv=forward(xf,random_graph['W'],random_graph['b'])
        m=metrics('fresh_random',chosen,hh,vv,fresh_h,fresh_v,W);m['draw']=draw;random.append(m)
    write(out/'random_graph_controls.csv',random)
    episode_ids={'heldout':[(split[int(i)]['seed'],split[int(i)]['episode_id']) for i in heldout],
        'fresh':[(r['seed'],r['episode_id']) for r in fr]}
    ci=bootstrap_results(populations,outputs,intervention_detail,args.bootstrap_draws,args.seed,episode_ids)
    write(out/'bootstrap_intervals.csv',ci)
    print('Intervention predictions, background tests and confidence intervals complete.',flush=True)
    plt.rcParams.update({'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','font.size':10,'pdf.fonttype':42})
    make_plots(out,selection,results,records,intervention_rows,background,chosen)
    examples=render_examples(out,xf,fr,fresh_h,fresh_v,outputs['fresh'],args.seed)
    validation=dict(full_critic_replay_max_error=err,finite_difference_chain_replay_max_error=chain_error,
        fresh_episodes_disjoint_original_bank_and_random_probe=True,all_edge_four_target_endpoint=True,
        critic_weights_unchanged=digest(out/'frozen_critic_weights.npz')==weight_hash,
        input_edits_validated=True,original_split_unchanged=True)
    summary=dict(protocol=protocol,selection=decision,fidelity=[r for r in results if r['k']==chosen],fresh_strata=strata,heldout_strata=heldout_strata,
        interventions=intervention_rows,bootstrap=ci,background=background,random_graph_controls=random,
        examples=examples,input_coverage=coverage,posthoc_input_diagnostics=diagnostics,
        validation=validation,elapsed_seconds=time.monotonic()-began,
        limitations=['The external set has 61 eligible episodes and five directly facing diamond; it has been analyzed previously and is not a new untouched test set.',
            'The bank may contain multiple states per episode; confidence intervals resample whole episodes and preserve all within-episode states.',
            'The expanded bank still comes from episodes in the original successful diamond population; it broadens states, not the eventual-success selection.',
            'Parents are real neurons; labels are response descriptions, not exclusive semantic identities.',
            'Finite-difference coefficients rank edges; circuit fidelity is determined by independent actual nonlinear execution.',
            'Non-target output contributions are a fixed discovery mean. Four neurons may capture interaction without explaining absolute value.',
            'No actor edits, policy learning, rollouts or claim of a complete environmental world model.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');write_report(out,summary)
    print(json.dumps(dict(selection=decision,fidelity=summary['fidelity'],validation=validation,elapsed=summary['elapsed_seconds']),indent=2),flush=True)


def interventions(out,rankings,graph,W,b,populations,outputs):
    tests=[]
    # Fixed upstream target choices; parent edges ranked on discovery only.
    for target,factor in [(159,'diamond'),(256,'pickaxe'),(334,'diamond'),(461,'diamond')]:
        selected=rankings[1][:4,target].tolist()
        tests.append((f'L1_to_L2_{target}_{factor}',1,target,selected,factor))
    for target,source,factor in [(198,256,'pickaxe'),(198,461,'diamond'),(198,159,'diamond'),(198,334,'diamond'),
                                 (147,int(rankings[2][0,147]),'diamond'),(11,int(rankings[2][0,11]),'diamond'),(439,int(rankings[2][0,439]),'diamond')]:
        tests.append((f'L2_{source}_to_L3_{target}_{factor}',2,target,[source],factor))
    # Test observed response at the input layer to the physical edits directly.
    test_rows=[];details=[]
    for population,(xx,truth,values) in populations.items():
        fullh,fullz,fullv=forward(xx,W,b);sparseh,sparsez,sparsev=outputs[population]
        for name,layer,target,selected,factor in tests:
            donor=3 if factor=='diamond' else 1  # both-present receiver
            actual_delta_z=((fullh[layer][donor][:,selected]-fullh[layer][0][:,selected])*W[layer][selected,target]).sum(1)
            sparse_delta_z=((sparseh[layer][donor][:,selected]-sparseh[layer][0][:,selected])*graph['W'][layer][selected,target]).sum(1)
            changed=fullh[layer+1][0].copy();changed[:,target]=np.tanh(fullz[layer][0,:,target]+actual_delta_z)
            predicted=sparseh[layer+1][0].copy();predicted[:,target]=np.tanh(sparsez[layer][0,:,target]+sparse_delta_z)
            for j in range(layer+1,3):
                changed=np.tanh(changed@W[j]+b[j]);predicted=np.tanh(predicted@graph['W'][j]+graph['b'][j])
            actual_full=((changed-fullh[3][0])@W[3]).reshape(-1)
            actual_path=((changed[:,TARGET]-fullh[3][0][:,TARGET])@W[3][TARGET]).reshape(-1)
            predicted_path=((predicted-sparseh[3][0])@graph['W'][3]).reshape(-1)
            mae=np.abs(predicted_path-actual_path).mean();scale=np.abs(actual_path).mean()
            nz=np.abs(actual_path)>1e-4
            row=dict(population=population,test=name,layer=layer+1,target=target,parents=';'.join(map(str,selected)),factor=factor,n=len(actual_path),
                actual_path_effect_mean=float(actual_path.mean()),predicted_effect_mean=float(predicted_path.mean()),actual_full_value_effect_mean=float(actual_full.mean()),
                effect_mae=float(mae),normalized_effect_mae=float(mae/max(scale,1e-8)),
                informative_states=int(nz.sum()),sign_agreement_pct=float(100*((predicted_path[nz]>0)==(actual_path[nz]>0)).mean()) if nz.any() else None,
                collateral_value_effect_mae=float(np.abs(actual_full-actual_path).mean()))
            test_rows.append(row)
            for i in range(len(actual_path)):
                details.append(dict(population=population,test=name,state_position=i,actual_path_effect=float(actual_path[i]),
                    predicted_effect=float(predicted_path[i]),actual_full_value_effect=float(actual_full[i])))
    write(out/'upstream_path_definitions.csv',[dict(name=n,layer=l+1,target=t,parents=';'.join(map(str,s)),factor=f) for n,l,t,s,f in tests])
    return test_rows,details


def input_diagnostics(out,graph,ref,W,b,old,fresh,fh,fv):
    observed=graph['masks'][0].any(axis=1)
    diamond=np.arange(63)*21+10
    coverage=[]
    counts=[]
    for name,x in [('original_bank',old[0]),('fresh',fresh[0])]:
        visible=x[:,:1323].reshape(-1,63,21)[:,:,10]>0
        counts.append(visible.sum(0).reshape(7,9))
        coverage.append(dict(population=name,n=len(x),retained_diamond_positions=np.flatnonzero(observed[diamond]).tolist(),
            states_with_any_retained_diamond_channel=int((visible&observed[diamond]).any(1).sum()),
            any_diamond_channel_retained_pct=float(100*(visible&observed[diamond]).any(1).mean()),
            iron_pickaxe_input_retained=bool(observed[1331])))
    # Post-hoc diagnosis, never used to change the primary selected graph.
    # Restore full original input edges for explicitly named input groups.
    variants=[('restore_all_diamond_channels',diamond),
        ('restore_all_diamond_and_stone_channels',np.r_[diamond,np.arange(63)*21+4]),
        ('restore_all_map_channels',np.arange(1323)),('restore_all_L1_inputs',np.arange(1345))]
    diagnostics=[];active=np.flatnonzero(graph['masks'][0].any(0))
    predictions={}
    for name,indices in variants:
        mask=graph['masks'][0].copy();mask[np.ix_(indices,active)]=1
        ww=list(graph['W']);bb=list(graph['b'])
        ww[0]=W[0]*mask;bb[0]=b[0]+ref[0]@(W[0]*(1-mask))
        h,z,v=forward(fresh,ww,bb)
        row=metrics(name,graph['k'],h,v,fh,fv,W)
        row['edges']=int(mask.sum()+graph['masks'][1].sum()+graph['masks'][2].sum()+len(TARGET))
        row['status']='Post-hoc structural diagnosis on the same fresh set; not an untouched generalization result.'
        diagnostics.append(row);predictions[name]=v
    write(out/'posthoc_input_diagnostics.csv',diagnostics)
    np.savez_compressed(out/'posthoc_input_predictions.npz',**predictions)
    (out/'input_coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    plt.rcParams.update({'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),constrained_layout=True)
    axes[0].imshow(observed[diamond].reshape(7,9),vmin=0,vmax=1,cmap='Blues')
    axes[0].set_title(f'Retained diamond inputs ({int(observed[diamond].sum())}/63)')
    pic=axes[1].imshow(counts[1],cmap='YlOrRd');axes[1].set_title(f'External diamond locations (n={len(fresh[0])})')
    for ax in axes:
        ax.scatter(4,3,marker='*',color='#16a34a',s=100,edgecolor='black');ax.set_xticks(range(9));ax.set_yticks(range(7));ax.set_xlabel('Observation column');ax.set_ylabel('Observation row')
    fig.colorbar(pic,ax=axes[1],label='Number of observed diamonds')
    save(fig,out,'input_position_coverage')
    return coverage,diagnostics


def background_analysis(out,rankings,ref,W,b,discovery,populations):
    core=[159,256,334,461];other=np.setdiff1d(np.arange(512),core);weights=W[2][:,198]
    records=[];summary=[]
    for name,(xx,_,_) in populations.items():
        h,z,v=forward(xx,W,b)
        support=h[2][:,:,other]@weights[other]+b[2][198]
        core_z=h[2][:,:,core]@weights[core]
        np.testing.assert_allclose(support+core_z,z[2][:,:,198],atol=1e-5)
        support_delta=support[0]-support[2]
        # Rank the omitted incoming signals by their actual discovery contrast.
        dh,_,_=forward(discovery,W,b)
        scores=(np.abs(dh[2][0]-dh[2][2])*np.abs(weights)).mean(0);scores[core]=-1
        ranked=np.argsort(-scores,kind='stable')[:16]
        summary.append(dict(population=name,n=xx.shape[1],support_mean={key:float(support[c].mean()) for c,key in enumerate('ABCD')},
            support_sd={key:float(support[c].std()) for c,key in enumerate('ABCD')},
            A_minus_neither_support_mean=float(support_delta.mean()),A_minus_neither_support_sd=float(support_delta.std()),
            discovery_ranked_background_parents=ranked.tolist()))
        for i in range(xx.shape[1]):
            row=dict(population=name,state_position=i)
            for c,key in enumerate('ABCD'):
                row[f'background_z_{key}']=float(support[c,i]);row[f'core_z_{key}']=float(core_z[c,i])
            records.append(row)
    write(out/'per_state_background.csv',records)
    # Input signal provenance for L1 ancestors of the four L2 core parents.
    provenance=[];dh,dz,_=forward(discovery,W,b)
    for target in core:
        for parent in rankings[1][:8,target]:
            for factor,donor in [('diamond',3),('pickaxe',1)]:
                delta_x=discovery[0]-discovery[donor]
                source_effect=delta_x*W[0][:,parent]
                weighted=(dh[1][0,:,parent]-dh[1][donor,:,parent])*W[1][parent,target]
                provenance.append(dict(L2_target=target,L1_parent=int(parent),factor=factor,
                    mean_input_preactivation_effect=float(source_effect.sum(1).mean()),
                    mean_L2_preactivation_contribution=float(weighted.mean()),
                    tool_inventory_input_weight=float(W[0][1331,parent]),
                    source='iron pickaxe input 1331' if factor=='pickaxe' else 'all visible diamond channels replaced by stone channels'))
    write(out/'input_signal_provenance.csv',provenance)
    return summary


def bootstrap_results(populations,outputs,details,draws,seed,episode_ids=None):
    rows=[];rng=np.random.default_rng(seed)
    for name,(xx,h,v) in populations.items():
        n=xx.shape[1]
        prediction=outputs[name][2];original=Q@v;predicted=Q@prediction
        data=np.column_stack([original,predicted,np.abs(original-predicted)])
        ids=episode_ids[name] if episode_ids is not None else list(range(n))
        lookup={};inverse=[]
        for eid in ids:
            if eid not in lookup:lookup[eid]=len(lookup)
            inverse.append(lookup[eid])
        inverse=np.array(inverse);sums=np.zeros((len(lookup),3));np.add.at(sums,inverse,data)
        counts=np.bincount(inverse);weights=rng.multinomial(len(lookup),np.full(len(lookup),1/len(lookup)),size=draws).astype(float)
        a=(weights@sums)/(weights@counts)[:,None]
        for key,data,point in [('original_mean_interaction',a[:,0],original.mean()),
                               ('predicted_mean_interaction',a[:,1],predicted.mean()),
                               ('interaction_mae',a[:,2],np.abs(original-predicted).mean())]:
            ci=np.quantile(data,[.025,.975]);rows.append(dict(population=name,metric=key,mean=float(point),low=float(ci[0]),high=float(ci[1]),n=n,episodes=len(lookup),draws=draws))
    return rows


def save(fig,out,name):
    fig.savefig(out/f'{name}.pdf');fig.savefig(out/f'{name}.png',dpi=240);plt.close(fig)


def make_plots(out,selection,results,records,interventions,background,chosen):
    from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
    chosen_row=next(r for r in selection if r['k']==chosen)
    fig,ax=plt.subplots(figsize=(12,3.3),constrained_layout=True);ax.set(xlim=(0,12),ylim=(0,3));ax.axis('off')
    with np.load(out/'selected_graph.npz') as saved_graph:
        diamond_positions=int(saved_graph['mask0'][np.arange(63)*21+10].any(1).sum())
    labels=[f'Symbolic input\n{int(chosen_row["input_nodes"])}/1,345 coordinates\nDiamond: {diamond_positions}/63 positions',
        f'Hidden layer 1\n{int(chosen_row["l1_nodes"])}/512 neurons\nOriginal tanh units',
        f'Hidden layer 2\n{int(chosen_row["l2_nodes"])}/512 neurons\n159, 256, 334, 461, …',
        'Hidden layer 3\n4/512 neurons\n147, 198, 11, 439','Scalar value\nRemaining output\nfixed to reference']
    for i,label in enumerate(labels):
        x=.1+i*2.4
        ax.add_patch(FancyBboxPatch((x,1.05),2.05,1.15,boxstyle='round,pad=.05',facecolor='#eff6ff' if i<4 else '#fef3c7',edgecolor='#2563eb'))
        ax.text(x+1.025,1.625,label,ha='center',va='center',fontweight='bold',fontsize=8)
        if i<4:ax.add_patch(FancyArrowPatch((x+2.12,1.625),(x+2.32,1.625),arrowstyle='-|>',mutation_scale=15,color='black'))
    ax.set_title('Input-to-value graph selected on discovery episodes',fontweight='bold')
    ax.text(.15,.55,f'{int(chosen_row["edges"]):,} original weighted connections retained; omitted messages use fixed discovery means.',fontweight='bold',fontsize=10)
    ax.text(.15,.18,'This is a broad sparse-network approximation. Counts describe the selected graph, not a proven minimal circuit.',fontweight='bold',fontsize=9)
    save(fig,out,'trace_overview')
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
    for name,color in [('discovery','#6b7280'),('heldout','#2563eb'),('fresh','#dc2626')]:
        rr=selection if name=='discovery' else [r for r in results if r['population']==name]
        for ax,metric,label in zip(axes,['interaction_normalized_mae','target_activation_mae','full_value_mae'],
                                  ['Interaction MAE / mean |original I|','Four-target activation MAE','Full scalar-value MAE']):
            ax.plot([r['edges'] for r in rr],[r[metric] for r in rr],marker='o',ms=3,color=color,label=name)
            ax.set(xscale='log',xlabel='Retained original weighted edges',ylabel=label);ax.grid(alpha=.2)
    axes[0].set_title('Predicting the interaction');axes[1].set_title('Predicting target features');axes[2].set_title('Explaining absolute value')
    axes[0].legend(frameon=False,prop={'weight':'bold','size':8});save(fig,out,'circuit_fidelity_frontier')
    fig,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
    for ax,name in zip(axes,['heldout','fresh']):
        rr=[r for r in records if r['population']==name]
        a=np.array([r['original_interaction'] for r in rr]);p=np.array([r['predicted_interaction'] for r in rr])
        ax.scatter(a,p,s=10,alpha=.5,color='#2563eb' if name=='heldout' else '#dc2626')
        lo=min(a.min(),p.min());hi=max(a.max(),p.max());ax.plot([lo,hi],[lo,hi],color='black',lw=.8)
        ax.set(xlabel='Original critic interaction',ylabel='Sparse-circuit prediction',title=f'{name.capitalize()}: n={len(rr)}, k={chosen}');ax.grid(alpha=.2)
    save(fig,out,'heldout_prediction_scatter')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for ax,name in zip(axes,['heldout','fresh']):
        rr=[r for r in interventions if r['population']==name]
        labels=[r['test'].replace('L1_to_L2_','L1→').replace('L2_','L2 ').replace('_to_L3_','→L3 ').replace('_diamond',' D').replace('_pickaxe',' P') for r in rr]
        y=np.arange(len(rr));ax.barh(y-.15,[r['actual_path_effect_mean'] for r in rr],height=.3,label='Actual intervention',color='#6b7280')
        ax.barh(y+.15,[r['predicted_effect_mean'] for r in rr],height=.3,label='Circuit prediction',color='#2563eb')
        ax.set_yticks(y,labels,fontsize=7);ax.invert_yaxis();ax.set_title(name.capitalize());ax.set_xlabel('Mean change in four-target value readout');ax.axvline(0,color='black',lw=.7)
    axes[0].legend(frameon=False,prop={'size':8,'weight':'bold'});save(fig,out,'intervention_predictions')
    fig,axes=plt.subplots(1,2,figsize=(9,3.7),constrained_layout=True)
    for ax,r in zip(axes,background):
        a=[r['support_mean'][key] for key in ['A','B','D','C']];s=[r['support_sd'][key] for key in ['A','B','D','C']]
        ax.bar(range(4),a,yerr=s,capsize=3,color=['#2563eb','#d97706','#dc2626','#0f766e'])
        ax.set_xticks(range(4),LABELS,rotation=20,ha='right');ax.set_ylabel('Other 508 inputs + bias (mean ± SD)');ax.set_title(r['population'].capitalize()+' background input')
    save(fig,out,'background_support')


def render_examples(out,xf,rows,truth,values,prediction,seed):
    from PIL import Image,ImageDraw,ImageFont,JpegImagePlugin,PdfImagePlugin
    from craftax.craftax_classic.constants import load_all_textures
    from concept_mapping.visualize_symbolic_trajectory import render_local_view
    dest=out/'examples';dest.mkdir(exist_ok=True);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',16)
    title=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',18)
    textures=load_all_textures(7);panel_textures=load_all_textures(28)
    selected=np.random.default_rng(seed+198).choice(len(rows),min(5,len(rows)),replace=False);sheets=[];selection=[]
    for number,i in enumerate(selected,1):
        panels=[]
        for position,c in enumerate(ORDER):
            view=render_local_view(xf[c,i],block_size=7,include_inventory=True,display_scale=4,textures=textures,panel_textures=panel_textures)
            panel=Image.new('RGB',(view.width,view.height+110),'white');panel.paste(view,(0,28));draw=ImageDraw.Draw(panel)
            draw.text((view.width/2,2),LABELS[position],anchor='mt',font=title,fill='black')
            labels=[f'value: {values[c,i]:.4f}',f'circuit: {prediction[2][c,i]:.4f}',f'h198: {truth[3][c,i,198]:+.3f} / {prediction[0][3][c,i,198]:+.3f}']
            for line,text in enumerate(labels):draw.text((3,view.height+33+line*24),text,font=font,fill='black' if line<2 else '#1d4ed8')
            panels.append(panel)
        w,h=panels[0].size;sheet=Image.new('RGB',(2*w+18,2*h+18),'white')
        for j,panel in enumerate(panels):sheet.paste(panel,((j%2)*(w+18),(j//2)*(h+18)))
        filename=f'example_{number:02d}.png';sheet.save(dest/filename);sheets.append(sheet)
        selection.append(dict(file=filename,state_position=int(i),**rows[i]))
    sheets[0].save(dest/'all_examples.pdf',save_all=True,append_images=sheets[1:]);write(dest/'selection.csv',selection)
    return selection


def write_report(out,r):
    s=r['selection'];p=r['protocol'];by_pop={a['population']:a for a in r['fidelity']}
    hold,external=by_pop['heldout'],by_pop['fresh']
    coverage={a['population']:a for a in r['input_coverage']}
    restoration=r['posthoc_input_diagnostics'][0]
    lines=['# Input-to-value diamond circuit experiment','',
        'Frozen regular feedforward PPO critic, seed 50, standard rewards, checkpoint 9,999,941,632. No training or new rollouts.','',
        '## Protocol','',r['protocol']['ranking'],'',r['protocol']['selection'],'',
        'All omitted incoming messages use fixed pooled discovery means. The four target output neurons are 147, 198, 11, 439; '
        'the remaining scalar-output contribution is fixed to its discovery mean. Candidate circuits execute real weighted tanh computations from the input. '
        'They never receive fresh-state original activations or recomputed finite-difference gains as prediction inputs.','',
        f'Bank: {p.get("bank_dir", "original diamond bank")}. '
        f'Discovery: {p["discovery_n"]:,} states from {p.get("discovery_episodes",2438):,} episodes. '
        f'Held-out: {p["heldout_n"]:,} states from {p.get("heldout_episodes",1045):,} episodes. '
        f'External: {p["fresh_n"]} episode-disjoint states from the previous external probe; these have been examined before. '
        'No episode crosses between discovery, held-out and external populations. Bootstrap intervals resample episodes and preserve their states.','',
        f'Chosen graph: k={s["chosen_k"]}; discovery criteria met: {s["discovery_criteria_met"]}. '
        f'{s["chosen_discovery_metrics"]["edges"]:,} retained weighted edges, including four scalar readout edges.','',
        '## Fidelity','',
        '| Population | N | Original mean I | Predicted mean I | Normalized interaction MAE | Target activation MAE | Full value MAE |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for a in r['fidelity']:lines.append(f'| {a["population"]} | {a["n"]} | {a["original_interaction_mean"]:.4f} | {a["predicted_interaction_mean"]:.4f} | {a["interaction_normalized_mae"]:.3f} | {a["target_activation_mae"]:.3f} | {a["full_value_mae"]:.3f} |')
    lines.extend(['','Normalized interaction MAE divides mean absolute prediction error by mean absolute original interaction. '
        'This evaluates individual-state prediction, rather than only recovering a mean that could hide cancellation. '
        'k=512 restores every upstream edge into the four target outputs and provides a ceiling for this four-output model.','',
        '## Main interpretation','',
        f'The selected graph has {s["chosen_discovery_metrics"]["edges"]:,} retained weighted connections. '
        f'It retains {hold["interaction_retained_pct"]:.2f}% of mean interaction on the held-out bank, '
        f'with normalized per-state interaction error {100*hold["interaction_normalized_mae"]:.2f}%. '
        f'On the external states it retains {external["interaction_retained_pct"]:.2f}%, with '
        f'normalized error {100*external["interaction_normalized_mae"]:.2f}%. '
        'These results describe this ranked sparse approximation; they do not establish that it is the smallest possible circuit.','',
        f'The selected graph retains diamond coordinates at {len(coverage["original_bank"]["retained_diamond_positions"])} '
        f'of 63 grid positions. At least one retained diamond coordinate is present in '
        f'{coverage["original_bank"]["any_diamond_channel_retained_pct"]:.2f}% of bank scenes and '
        f'{coverage["fresh"]["any_diamond_channel_retained_pct"]:.2f}% of external scenes. '
        'The center tile cannot naturally hold diamond while occupied by the player.','',
        '### Post-hoc input-coverage diagnosis','',
        'The structural restoration checks were introduced during the prior narrow-bank study and remain diagnostic. '
        'They do not alter primary graph selection. The external set is not an untouched replication population. '
        f'Restoring original connections from all diamond-position inputs changes external interaction MAE from '
        f'{external["interaction_mae"]:.4f} to {restoration["interaction_mae"]:.4f}; '
        f'the restored mean interaction is {restoration["predicted_interaction_mean"]:.4f}, compared with '
        f'the full critic’s {external["original_interaction_mean"]:.4f}.','',
        '| Diagnostic restoration | Connections | Fresh interaction MAE | Predicted mean I |',
        '| --- | ---: | ---: | ---: |']+
        [f'| {a["population"]} | {a["edges"]:,} | {a["interaction_mae"]:.4f} | {a["predicted_interaction_mean"]:.4f} |' for a in r['posthoc_input_diagnostics']]+['',
        '## Intervention predictions','',
        'Tests route same-state donor messages only on specified incoming connections, then execute the remaining layers. '
        'Four discovery-ranked incoming L1 messages are tested into each of L2 parents 159/256/334/461. '
        'Single L2 messages are also tested into all four target L3 neurons. '
        'The fixed sparse circuit predicts changes without fitting to test effects. '
        'Both the actual four-target readout effect and the full-critic effect are saved: discrepancies reveal collateral computation outside the proposed circuit.','',
        '## Background input','',
        'For neuron 198, background is the actual weighted sum of the other 508 L2 inputs plus the bias. '
        'Its means and variation are reported separately by condition and population. '
        'Input-signal provenance identifies early neurons receiving the literal iron-pickaxe slot and diamond-to-stone edits; '
        'those edited-input contributions are exact, but their broader semantic labels remain hypotheses. '
        'The other 508 L2 inputs are not simply noise: their contribution to neuron 198 varies with state and condition. '
        'The per-state background table and plot expose that variation instead of treating it as a harmless constant.','',
        '## Artifacts','',
        '- [Trace overview](trace_overview.pdf)',
        '- [Fidelity frontier](circuit_fidelity_frontier.pdf)',
        '- [Fresh and held-out prediction scatter](heldout_prediction_scatter.pdf)',
        '- [Intervention predictions](intervention_predictions.pdf)',
        '- [Background support](background_support.pdf)',
        '- [Input-position coverage](input_position_coverage.pdf), [post-hoc input diagnostics](posthoc_input_diagnostics.csv)',
        '- [Five fresh-state grids](examples/all_examples.pdf)',
        '- [Fixed selected edges](selected_edges.csv), [input provenance](input_signal_provenance.csv)',
        '- [Per-state predictions](per_state_predictions.csv), [per-state interventions](per_state_interventions.csv)',
        '- [Episode-bootstrap intervals](bootstrap_intervals.csv), [held-out strata](heldout_strata.csv), [external strata](fresh_strata.csv), [full report](summary.json)','',
        '## Limits','']+[f'- {x}' for x in r['limitations']]+['',
        'Reproduce in jax-v100-cuda12:','', '```bash',
        f'OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/mpl-input-circuit python concept_mapping/trace_diamond_input_to_value.py --bank-dir {p.get("bank_dir", "concept_mapping/runs/diamond_collectability_10b_10000episodes_all_visible_stone")} --out {out}',
        '```',''])
    (out/'README.md').write_text('\n'.join(lines))


if __name__=='__main__':main()

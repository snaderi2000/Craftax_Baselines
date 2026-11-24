"""Extend the partial neuron-198 circuit on fit/dev data, then lock it.

Fresh rollout evaluation is a separate phase and cannot change the selected
mask. Candidate additions are actual nonlinear restorations of omitted paths.
"""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import concept_mapping.prune_neuron198_paths as cp
import concept_mapping.prune_neuron198_joint as joint

OUT=cp.ROOT/'neuron198_extension_validation'
OLD=cp.OUT/'joint_refinement'
SEED=50198
ROLLOUT_SEED=19811026


def strict(m):
    return m['activation_mae']<=.03 and m['recovery_error']<=.06 and 95<=m['recovery_pct']<=105 and m['residual_error']<=.10 and abs(m['residual_pct'])<=5


def load():
    with np.load(cp.SOURCE/'frozen_critic_weights.npz') as d:W=[d[f'W{j}'] for j in range(4)];b=[d[f'b{j}'] for j in range(4)]
    cp.EVAL_WEIGHTS=W
    return W,b


def contexts(out,label,model,indices,metadata):
    original=cp.Q@(model.clean-model.donor);predicted=cp.Q@(np.tanh(model.zr)-model.donor)
    error=np.abs(original-predicted);rows=[]
    for pos,i in enumerate(indices):
        r=metadata[int(i)]
        health=int(r['health']);ach=int(r['achievement_count']);step=int(r['timestep'])
        rows.append(dict(state_index=int(i),stratum=r['stratum'],health=health,achievement_count=ach,timestep=step,
            health_group='1_3' if health<=3 else '4_6' if health<=6 else '7_9',
            achievement_group='up_to_15' if ach<=15 else '16_plus',
            step_group='up_to_500' if step<=500 else '501_1500' if step<=1500 else '1501_plus',
            original_interaction=float(original[pos]),prediction=float(predicted[pos]),absolute_error=float(error[pos])))
    cp.write(out/f'{label}_state_errors.csv',rows)
    groups=[]
    for dimension in ['stratum','health_group','achievement_group','step_group']:
        for value in sorted({r[dimension] for r in rows}):
            rr=[r for r in rows if r[dimension]==value];scale=np.mean([abs(r['original_interaction']) for r in rr])
            groups.append(dict(population=label,dimension=dimension,group=value,n_states=len(rr),
                mean_absolute_error=float(np.mean([r['absolute_error'] for r in rr])),
                normalized_error=float(np.mean([r['absolute_error'] for r in rr])/max(scale,1e-8))))
    cp.write(out/f'{label}_error_contexts.csv',groups)
    return groups


def fit(out):
    out.mkdir(parents=True,exist_ok=True);began=time.monotonic();W,b=load()
    with np.load(cp.BANK/'base_states.npz') as d:x=cp.branches(d['observations'])
    rows=cp.read(cp.OUT/'pruning_episode_split.csv');metadata=cp.read(cp.BANK/'state_selection.csv')
    fit_idx=np.array([i for i,r in enumerate(rows) if r['pruning_split']=='fit']);dev_idx=np.array([i for i,r in enumerate(rows) if r['pruning_split']=='dev'])
    full=cp.forward(x,W,b)
    with np.load(OLD/'path_mask.npz') as d:parents=d['L2_parents'];mask=d['L1_to_L2_mask']
    baseline_mask=(parents.copy(),mask.copy())
    models={name:cp.PathModel(full,W,b,idx,'joint',parents,mask) for name,idx in [('fit',fit_idx),('dev',dev_idx)]}
    protocol=dict(checkpoint=9999941632,run='wandb/run-20260630_214658-m0mw4end',target=198,
        motivation='Explain omitted-path errors in the previous six-parent partial circuit before testing new episodes.',
        fit_and_dev='Same episode split as previous pruning: 1,950 fit, 488 development. No old held-out or external responses select additions.',
        primary_test_criteria=json.loads((cp.OUT/'protocol.json').read_text())['criteria'],
        tighter_selection_margin=dict(activation_mae=.03,recovery_error=.06,recovery_pct=[95,105],necessity_error=.10,necessity_mean_residual_pct=5),
        additions='Restore either a missing L2 parent with all original incoming L1 messages, or all omitted incoming L1 messages of a retained L2 parent. Rank on fit states; accept only if fit and dev nonlinear loss improve.',
        budget=dict(max_addition_rounds=12,max_L2_parents=16,screen_states=512,dev_candidates_per_round=20),
        stop='Stop at tighter fit/dev margins, no addition improving both losses by 0.005, or budget. Then causally prune only while retaining tighter margins.',
        fresh_validation=dict(episodes=512,seed=ROLLOUT_SEED,max_steps=10000,policy='frozen categorical standard-reward feedforward PPO',
            collection='Begins only after LOCKED_SELECTION.json is saved. One state per available spatial stratum per episode; no future-success filter.',
            sampling_seed=ROLLOUT_SEED+1),
        no_policy_or_critic_training=True,seed=SEED)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    original_contexts=[]
    for name,idx in [('fit',fit_idx),('dev',dev_idx)]:original_contexts.extend(contexts(out,'original_'+name,models[name],idx,metadata))
    # Decompose missing target preactivation exactly into omitted L2 inputs and
    # incomplete upstream inputs of retained L2 parents. This is descriptive;
    # actual intervention losses decide which groups to add.
    decomposition=[]
    for name,idx in [('fit',fit_idx),('dev',dev_idx)]:
        m=models[name];other=np.setdiff1d(np.arange(512),parents)
        missing_direct=(full[0][2][:,idx][:,:,other]-full[0][2][2,idx][:,other][None])@W[2][other,198]
        missing_upstream=((m.h2r-np.tanh(m.a2))*m.w[parents]).sum(-1)
        residual=m.zclean-m.zr
        np.testing.assert_allclose(missing_direct+missing_upstream,residual,atol=1e-5,rtol=1e-5)
        for c,key in enumerate('ABCD'):
            decomposition.append(dict(population=name,condition=key,mean_omitted_L2=float(missing_direct[c].mean()),
                mean_upstream_truncation=float(missing_upstream[c].mean()),mean_absolute_residual=float(np.abs(residual[c]).mean()),
                mean_absolute_omitted_L2=float(np.abs(missing_direct[c]).mean()),mean_absolute_upstream=float(np.abs(missing_upstream[c]).mean())))
    cp.write(out/'error_decomposition.csv',decomposition)
    rng=np.random.default_rng(SEED+198);screen=np.sort(rng.choice(len(fit_idx),min(512,len(fit_idx)),replace=False))
    screening=[];trajectory=[]
    def record(kind,parent):
        trajectory.append(dict(step=len(trajectory),kind=kind,parent=parent,parents=parents.tolist(),upstream_edges=int(mask.sum()),
            fit=models['fit'].metric(),dev=models['dev'].metric()))
    record('original',-1)
    for step in range(12):
        if strict(models['fit'].metric()) and strict(models['dev'].metric()):break
        candidate=[];fm=models['fit']
        for parent in range(512):
            if parent in parents:
                j=int(np.flatnonzero(parents==parent)[0])
                if mask[j].all():continue
                delta_r=fm.w[parent]*(fm.h2r[:,:,j]-np.tanh(fm.a2[:,:,j]))
                delta_n=fm.w[parent]*(fm.h2d[:,:,j]-np.tanh(fm.n2[:,:,j]))
                kind='complete_upstream'
            else:
                if len(parents)>=16:continue
                delta_r=(full[0][2][:,fit_idx,parent]-full[0][2][2,fit_idx,parent][None])*W[2][parent,198]
                delta_n=-delta_r;kind='add_parent'
            mm=cp.measure(fm.zr[:,screen]+delta_r[:,screen],fm.zn[:,screen]+delta_n[:,screen],fm.clean[:,screen],fm.donor[:,screen])
            candidate.append((cp.cost(mm),parent,kind))
        shortlist=sorted(candidate)[:20];eligible=[]
        for rank,(_,parent,kind) in enumerate(shortlist,1):
            assessments={}
            for population,idx in [('fit',fit_idx),('dev',dev_idx)]:
                model=models[population]
                if kind=='add_parent':
                    delta_r=(full[0][2][:,idx,parent]-full[0][2][2,idx,parent][None])*W[2][parent,198];delta_n=-delta_r
                else:
                    j=int(np.flatnonzero(parents==parent)[0]);delta_r=model.w[parent]*(model.h2r[:,:,j]-np.tanh(model.a2[:,:,j]));delta_n=model.w[parent]*(model.h2d[:,:,j]-np.tanh(model.n2[:,:,j]))
                assessments[population]=cp.measure(model.zr+delta_r,model.zn+delta_n,model.clean,model.donor)
            fit_gain=cp.cost(models['fit'].metric())-cp.cost(assessments['fit']);dev_gain=cp.cost(models['dev'].metric())-cp.cost(assessments['dev'])
            screening.append(dict(round=step,parent=parent,kind=kind,fit_rank=rank,fit_gain=fit_gain,dev_gain=dev_gain,
                fit_error=assessments['fit']['recovery_error'],dev_error=assessments['dev']['recovery_error']))
            if fit_gain>.005 and dev_gain>.005 and assessments['fit']['residual_error']<=.15 and assessments['dev']['residual_error']<=.15:
                eligible.append((cp.cost(assessments['dev']),parent,kind))
        if not eligible:
            print('No further fit/dev-supported addition.',flush=True);break
        _,parent,kind=min(eligible)
        if kind=='add_parent':parents=np.r_[parents,parent];mask=np.vstack([mask,np.ones(512,bool)])
        else:mask[int(np.flatnonzero(parents==parent)[0])]=True
        models={name:cp.PathModel(full,W,b,idx,'joint',parents,mask) for name,idx in [('fit',fit_idx),('dev',dev_idx)]}
        record(kind,parent);print(f'Accepted {kind} {parent}: fit {models["fit"].metric()}, dev {models["dev"].metric()}',flush=True)
    margin_met=all(strict(m.metric()) for m in models.values());history=[]
    if margin_met:
        original_pass=cp.passes
        try:
            cp.passes=strict
            mask,pruning=cp.prune(models['fit'],models['dev'],SEED+2,'extension_L1_to_L2',history)
        finally:cp.passes=original_pass
    else:pruning=dict(skipped=True,reason='Tighter selection margins not reached; preserve the best fit/dev additions.')
    live=mask.any(1);parents=parents[live];mask=mask[live];l1=np.flatnonzero(mask.any(0))
    inputs=np.flatnonzero(np.any(x[:,np.r_[fit_idx,dev_idx]]!=x[2,np.r_[fit_idx,dev_idx]][None],axis=(0,1)))
    np.savez_compressed(out/'path_mask.npz',L2_parents=parents,L1_to_L2_mask=mask,L1_parents=l1,input_coordinates=inputs)
    selection=dict(L2_parents=parents.tolist(),L1_parents=l1.tolist(),input_coordinates=inputs.tolist(),input_edges=len(inputs)*len(l1),
        L1_to_L2_edges=int(mask.sum()),L2_to_198_edges=len(parents),output_edges=1,
        original_parents=baseline_mask[0].tolist(),original_upstream_edges=int(baseline_mask[1].sum()),
        addition_trajectory=trajectory,tighter_margin_reached_before_pruning=margin_met,pruning=pruning)
    (out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n');cp.write(out/'candidate_additions.csv',screening)
    if history:cp.write(out/'pruning_history.csv',history)
    final_metrics={}
    for name,idx in [('fit',fit_idx),('dev',dev_idx)]:
        model=cp.PathModel(full,W,b,idx,'joint',parents,mask);final_metrics[name]=model.metric();contexts(out,'extended_'+name,model,idx,metadata)
    lock=dict(mask_sha256=cp.digest(out/'path_mask.npz'),selection_sha256=cp.digest(out/'selection.json'),
        critic_weights_sha256=cp.digest(cp.SOURCE/'frozen_critic_weights.npz'),script_sha256=cp.digest(Path(__file__)),
        selected_using='fit/development episodes only; no new validation episodes collected yet',
        final_fit_dev_metrics=final_metrics,seed=ROLLOUT_SEED,requested_fresh_episodes=512,
        elapsed_seconds=time.monotonic()-began)
    (out/'LOCKED_SELECTION.json').write_text(json.dumps(lock,indent=2)+'\n')
    (out/'fit_summary.json').write_text(json.dumps(dict(protocol=protocol,selection=selection,lock=lock,contexts=original_contexts,error_decomposition=decomposition),indent=2)+'\n')
    plot_fit(out,trajectory,final_metrics)
    print(json.dumps(dict(selection={k:v for k,v in selection.items() if k not in ['L1_parents','input_coordinates','addition_trajectory']},lock=lock),indent=2),flush=True)


def plot_fit(out,trajectory,final):
    plt.rcParams.update({'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(9,3.7),constrained_layout=True)
    for ax,metric,title in zip(axes,['recovery_error','activation_mae'],['Interaction recovery error','Activation reconstruction error']):
        for population,color in [('fit','#6b7280'),('dev','#2563eb')]:
            ax.plot(range(len(trajectory)),[r[population][metric] for r in trajectory],marker='o',color=color,label=population)
            ax.scatter(len(trajectory),final[population][metric],color=color,marker='s',s=45)
        ax.set(xlabel='Accepted additions; final square = pruned mask',ylabel=metric,title=title);ax.grid(alpha=.2)
    axes[0].legend(frameon=False,prop={'weight':'bold','size':8});cp.save(fig,out,'fit_dev_extension')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['fit','evaluate'],required=True);p.add_argument('--out',type=Path,default=OUT);p.add_argument('--fresh-dir',type=Path)
    args=p.parse_args()
    if args.phase=='fit':fit(args.out)
    else:evaluate_fresh(args.out,args.fresh_dir)


def evaluate_fresh(out,fresh_dir):
    if fresh_dir is None:
        raise ValueError('--fresh-dir is required for the locked evaluation')
    out=Path(out);fresh_dir=Path(fresh_dir)
    lock=json.loads((out/'LOCKED_SELECTION.json').read_text())
    for key,path in [('mask_sha256',out/'path_mask.npz'),
                     ('selection_sha256',out/'selection.json'),
                     ('critic_weights_sha256',cp.SOURCE/'frozen_critic_weights.npz')]:
        if cp.digest(path)!=lock[key]:raise RuntimeError(f'Locked artifact changed: {path}')
    meta=json.loads((fresh_dir/'dataset_metadata.json').read_text())
    expected={'run_path':'wandb/run-20260630_214658-m0mw4end','checkpoint':9999941632,
              'policy_sampling':'categorical sampling with saved seed','seed':ROLLOUT_SEED,
              'requested_episodes':512,'max_steps_per_episode':10000}
    for key,value in expected.items():
        if meta.get(key)!=value:raise RuntimeError(f'Unexpected collection metadata: {key}={meta.get(key)}')
    if (fresh_dir/'dataset_metadata.json').stat().st_mtime < (out/'LOCKED_SELECTION.json').stat().st_mtime:
        raise RuntimeError('Fresh collection metadata predates the locked selection')
    files=sorted((fresh_dir/'episodes').glob('episode_*.npz'))
    ids=[int(path.stem.split('_')[-1]) for path in files]
    if ids!=list(range(512)):raise RuntimeError(f'Expected episodes 0..511, got {len(ids)} files')
    strata=['adjacent_facing','adjacent_turn_required','distance_2','distance_3_4','distance_5_7']
    observations=[];rows=[];episode_outcomes=[]
    for path,eid in zip(files,ids):
        with np.load(path) as d:
            inv=d['inventory'];blocks=d['visible_blocks'];ach=d['achievements']
            n=len(blocks)
            if any(len(d[key])!=n for key in ['observations','inventory','achievements','sleeping','direction','intrinsics','dones']):
                raise RuntimeError(f'Inconsistent episode arrays: {path}')
            if not 1<=n<=10000:raise RuntimeError(f'Invalid episode length: {path}')
            coords=np.argwhere(blocks==10);distance=np.full(n,99,dtype=int)
            if len(coords):np.minimum.at(distance,coords[:,0],np.abs(coords[:,1:]-[3,4]).sum(1))
            directions=d['direction'].astype(int)
            dr=np.array([0,0,-1,1])[directions];dc=np.array([-1,1,0,0])[directions]
            facing=blocks[np.arange(n),3+dr,4+dc]==10
            eligible=(inv[:,8]>0)&(d['sleeping']==0)&(ach[:,19]==0)&(distance<=7)
            masks=[eligible&(distance==1)&facing,eligible&(distance==1)&~facing,
                   eligible&(distance==2),eligible&(distance>=3)&(distance<=4),
                   eligible&(distance>=5)&(distance<=7)]
            rng=np.random.default_rng(np.random.SeedSequence([ROLLOUT_SEED+1,ROLLOUT_SEED,eid]))
            for stratum,mask in zip(strata,masks):
                candidates=np.flatnonzero(mask)
                if not len(candidates):continue
                t=int(rng.choice(candidates));state_index=len(rows)
                obs=d['observations'][t].astype(np.float32)
                # This checks that saved observations and derived fields agree.
                if not np.isclose(obs[1331]*10,inv[t,8],atol=1e-3):raise RuntimeError(f'Observation/inventory mismatch: {path}:{t}')
                rows.append(dict(state_index=state_index,seed=ROLLOUT_SEED,episode_id=eid,timestep=t,
                    stratum=stratum,distance_to_diamond=int(distance[t]),diamond_in_front=int(facing[t]),
                    facing_direction=int(directions[t]),visible_diamonds=int((blocks[t]==10).sum()),
                    health=int(d['intrinsics'][t,0]),food=int(d['intrinsics'][t,1]),
                    water=int(d['intrinsics'][t,2]),energy=int(d['intrinsics'][t,3]),
                    achievement_count=int(ach[t].sum()),iron_pickaxe_count=int(inv[t,8]),
                    eligible_timesteps_in_stratum=len(candidates),source_episode=str(path)))
                observations.append(obs)
            diamond_collected=bool(d['terminal_achievements'][19] or ach[:,19].any())
            episode_outcomes.append(dict(seed=ROLLOUT_SEED,episode_id=eid,length=n,
                terminated=bool(d['dones'][-1]),diamond_collected=diamond_collected,
                eligible_strata=sum(bool(mask.any()) for mask in masks)))
    if not rows:raise RuntimeError('No eligible fresh diamond states')
    factual=np.stack(observations)
    np.savez_compressed(out/'fresh_validation_base_states.npz',observations=factual,
        episode_id=np.array([r['episode_id'] for r in rows]),
        timestep=np.array([r['timestep'] for r in rows]))
    cp.write(out/'fresh_validation_selection.csv',rows)
    cp.write(out/'fresh_episode_outcomes.csv',episode_outcomes)
    cohort=dict(collection_seed=ROLLOUT_SEED,sampling_seed=ROLLOUT_SEED+1,
        episodes_collected=512,episodes_with_eligible_states=len({r['episode_id'] for r in rows}),
        states=len(rows),strata={s:sum(r['stratum']==s for r in rows) for s in strata},
        diamond_collection_episodes=sum(r['diamond_collected'] for r in episode_outcomes),
        eligible_episodes_eventually_collecting_diamond=sum(r['diamond_collected'] and r['eligible_strata']>0 for r in episode_outcomes),
        capped_episodes=sum(r['length']==10000 for r in episode_outcomes),
        selection='One uniformly sampled natural pre-first-diamond state per available distance/facing stratum per episode, independent of eventual outcome and critic response.')
    (out/'fresh_validation_cohort.json').write_text(json.dumps(cohort,indent=2)+'\n')
    W,b=load();branches=cp.branches(factual);full=cp.forward(branches,W,b)
    index=np.arange(len(rows))
    results={};records={}
    for name,path in [('previous',OLD/'path_mask.npz'),('extended',out/'path_mask.npz')]:
        with np.load(path) as d:parents=d['L2_parents'];mask=d['L1_to_L2_mask']
        model=cp.PathModel(full,W,b,index,'joint',parents,mask)
        result,record=cp.evaluate(model,full,index,'joint',rows,10000,ROLLOUT_SEED+2)
        result.update(model=name,L2_parents=parents.tolist(),L1_to_L2_edges=int(mask.sum()))
        results[name]=result;records[name]=record
        cp.write(out/f'fresh_{name}_per_state.csv',record)
        print(f'{name} fresh: recovery error={result["recovery_error"]:.4f}, '
              f'activation MAE={result["activation_mae"]:.4f}, '
              f'recovery={result["recovery_pct"]:.2f}%, pass={result["passes"]}',flush=True)
    target=np.array([r['original_h_interaction'] for r in records['extended']])
    old=np.array([r['restored_h_interaction'] for r in records['previous']])
    new=np.array([r['restored_h_interaction'] for r in records['extended']])
    np.testing.assert_allclose(target,[r['original_h_interaction'] for r in records['previous']])
    scale=max(float(np.abs(target).mean()),1e-8)
    old_err=np.abs(old-target);new_err=np.abs(new-target);improvement=old_err-new_err
    grouped=[(r['seed'],r['episode_id']) for r in rows]
    boot=cp.cluster_bootstrap(np.column_stack([old_err,new_err,improvement]),grouped,10000,ROLLOUT_SEED+3)
    ci=np.quantile(boot[:,2],[.025,.975])
    comparison=dict(mean_original_absolute_error=float(old_err.mean()),mean_extended_absolute_error=float(new_err.mean()),
        mean_absolute_error_reduction=float(improvement.mean()),error_reduction_ci95=ci.tolist(),
        normalized_error_reduction=float(improvement.mean()/scale),
        normalized_error_reduction_ci95=(ci/scale).tolist(),
        fraction_states_improved=float((improvement>0).mean()),
        fraction_states_worsened=float((improvement<0).mean()),
        episode_bootstrap_draws=10000,bootstrap_unit='episode')
    (out/'fresh_validation_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    by_stratum=[]
    for s in strata:
        ix=np.array([i for i,r in enumerate(rows) if r['stratum']==s])
        by_stratum.append(dict(stratum=s,n_states=len(ix),n_episodes=len({rows[i]['episode_id'] for i in ix}),
            previous_normalized_error=float(old_err[ix].mean()/max(np.abs(target[ix]).mean(),1e-8)) if len(ix) else None,
            extended_normalized_error=float(new_err[ix].mean()/max(np.abs(target[ix]).mean(),1e-8)) if len(ix) else None,
            fraction_improved=float((improvement[ix]>0).mean()) if len(ix) else None))
    cp.write(out/'fresh_validation_by_stratum.csv',by_stratum)
    # An independent dense message replay checks the optimized path model on
    # fresh states, including the literal changed input coordinates.
    with np.load(out/'path_mask.npz') as d:
        parents=d['L2_parents'];mask=d['L1_to_L2_mask'];inputs=d['input_coordinates'];l1=d['L1_parents']
    check=np.arange(min(64,len(rows)))
    donor_z1=full[1][0][2,check][:,l1]
    dx=branches[:,check][:,:,inputs]-branches[2,check][None,:,inputs]
    replay_l1=np.tanh(donor_z1[None]+dx@W[0][np.ix_(inputs,l1)])
    np.testing.assert_allclose(replay_l1,full[0][1][:,check][:,:,l1],atol=2e-5,rtol=2e-5)
    selected_mask=mask[:,l1].T
    donor_z2=full[1][1][2,check][:,parents]
    donor_h1=full[0][1][2,check][:,l1]
    replay_z2=donor_z2[None]+(replay_l1-donor_h1[None])@(W[1][np.ix_(l1,parents)]*selected_mask)
    replay_h2=np.tanh(replay_z2)
    donor_h2=full[0][2][2,check][:,parents]
    replay_z3=full[1][2][2,check,198][None]+(replay_h2-donor_h2[None])@W[2][parents,198]
    independent=cp.PathModel(full,W,b,check,'joint',parents,mask)
    np.testing.assert_allclose(replay_z3,independent.zr,atol=2e-5,rtol=2e-5)
    validation=dict(mask_unchanged=cp.digest(out/'path_mask.npz')==lock['mask_sha256'],
        selection_unchanged=cp.digest(out/'selection.json')==lock['selection_sha256'],
        critic_weights_unchanged=cp.digest(cp.SOURCE/'frozen_critic_weights.npz')==lock['critic_weights_sha256'],
        all_512_episode_ids_present=ids==list(range(512)),
        sampled_state_unique=len({(r['episode_id'],r['timestep']) for r in rows})==len(rows),
        all_selected_pickaxe_present=bool((factual[:,1331]>0).all()),
        all_pre_first_diamond=True,branch_edits_validated=True,
        independent_dense_path_replay_64_states=True)
    summary=dict(cohort=cohort,results=results,comparison=comparison,by_stratum=by_stratum,
        validation=validation,selection_lock=lock,
        limitation='One trained critic and a conditional path circuit. Fresh sampling requires visible diamond and an iron pickaxe. Although eventual success was not used for selection, every eligible episode in this collection eventually obtained diamond, so failed-opportunity episodes are untested. Surrounding critic computation is preserved.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot_fresh(out,target,old,new,old_err,new_err,rows,results,by_stratum)
    plot_conditions(out,records['extended'])
    (out/'README.md').write_text('\n'.join([
        '# Locked neuron-198 circuit extension and fresh validation','',
        'The standard-reward 10B feedforward PPO policy and critic are frozen. The extension adds L2 parents 55 and 8 to the earlier six-parent circuit and prunes on fit/development episodes only. Its mask was locked before collecting the fresh episodes.','',
        f'Fresh collection: 512 complete episodes, seed {ROLLOUT_SEED}, checkpoint 9,999,941,632. {len(rows)} selected natural states from {cohort["episodes_with_eligible_states"]} episodes; one per available spatial stratum per episode. Selection does not depend on eventual diamond success or value.','',
        f'All {cohort["episodes_with_eligible_states"]} eligible episodes eventually collected diamond. This cohort therefore tests spatial and inventory context variation among successful episodes, not failed diamond opportunities.','',
        '| Circuit | L1→L2 edges | Target recovery | Normalized interaction error | Activation MAE | Original criteria pass |',
        '| --- | ---: | ---: | ---: | ---: | --- |',
        *[f'| {name} | {results[name]["L1_to_L2_edges"]} | {results[name]["recovery_pct"]:.2f}% | {results[name]["recovery_error"]:.3f} | {results[name]["activation_mae"]:.3f} | {results[name]["passes"]} |' for name in ['previous','extended']],
        '',f'Paired episode-bootstrap mean absolute interaction-error reduction: {comparison["mean_absolute_error_reduction"]:.4f} (95% CI {ci[0]:.4f}, {ci[1]:.4f}).',
        '',f'[Fresh validation figure](fresh_validation.pdf) · [Four-condition activations](fresh_condition_activations.pdf) · [Per-stratum results](fresh_validation_by_stratum.csv) · [Summary JSON](summary.json) · [Selected states](fresh_validation_selection.csv) · [Locked selection](LOCKED_SELECTION.json).',
        '',summary['limitation'],'']))
    print(json.dumps(dict(cohort=cohort,results={k:{x:v for x,v in r.items() if x in ['recovery_pct','recovery_error','activation_mae','residual_error','passes']} for k,r in results.items()},comparison=comparison,validation=validation),indent=2),flush=True)


def plot_fresh(out,target,old,new,old_err,new_err,rows,results,by_stratum):
    plt.rcParams.update({'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),constrained_layout=True)
    colors={'previous':'#6b7280','extended':'#dc2626'}
    for name in ['previous','extended']:
        result=results[name]
        axes[0].bar(name,result['recovery_error'],color=colors[name])
        axes[0].text(name,result['recovery_error']+.003,f'{result["recovery_error"]:.3f}',ha='center',fontweight='bold')
    axes[0].axhline(.10,color='black',ls='--',lw=1,label='original criterion')
    axes[0].set(ylabel='Normalized interaction error',title=f'Fresh {results["extended"]["n_episodes"]}-episode probe (512 collected)',ylim=(0,max(.13,max(r['recovery_error'] for r in results.values())*1.25)))
    axes[0].legend(frameon=False,prop={'weight':'bold','size':7})
    lim=(min(target.min(),old.min(),new.min()),max(target.max(),old.max(),new.max()))
    axes[1].plot(lim,lim,color='black',lw=1,ls='--')
    axes[1].scatter(target,old,s=7,alpha=.2,color=colors['previous'],label='Previous')
    axes[1].scatter(target,new,s=7,alpha=.35,color=colors['extended'],label='Extended')
    axes[1].set(xlabel='Full neuron-198 interaction',ylabel='Restored interaction',title='Statewise replay')
    axes[1].legend(frameon=False,prop={'weight':'bold','size':8})
    x=np.arange(len(by_stratum));width=.39
    axes[2].bar(x-width/2,[r['previous_normalized_error'] or 0 for r in by_stratum],width,color=colors['previous'],label='Previous')
    axes[2].bar(x+width/2,[r['extended_normalized_error'] or 0 for r in by_stratum],width,color=colors['extended'],label='Extended')
    axes[2].set_xticks(x,['Facing','Turn','2','3–4','5–7'],rotation=20)
    axes[2].set(xlabel='Diamond distance / facing',ylabel='Normalized interaction error',title='Spatial generalization')
    axes[2].legend(frameon=False,prop={'weight':'bold','size':8})
    cp.save(fig,out,'fresh_validation')


def plot_conditions(out,records):
    order=['A','B','D','C']
    matrix=np.array([
        [np.mean([r[f'h198_{key}'] for r in records]) for key in order],
        [np.mean([r[f'h198_{key}_restored'] for r in records]) for key in order],
        [np.mean([r[f'h198_{key}_removed'] for r in records]) for key in order],
    ])
    fig,ax=plt.subplots(figsize=(6.5,3),constrained_layout=True)
    pic=ax.imshow(matrix,vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto')
    ax.set_xticks(range(4),['Diamond + pickaxe','Diamond only','Pickaxe only','Neither'],rotation=15,ha='right')
    ax.set_yticks(range(3),['Full critic','Paths restored','Paths blocked'])
    ax.set_title('Neuron 198: fresh natural-state counterfactuals',fontweight='bold')
    for i in range(3):
        for j in range(4):ax.text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',fontweight='bold',
                                   color='white' if abs(matrix[i,j])>.6 else 'black')
    fig.colorbar(pic,ax=ax,label='Mean neuron activation',shrink=.8)
    cp.save(fig,out,'fresh_condition_activations')


if __name__=='__main__':main()

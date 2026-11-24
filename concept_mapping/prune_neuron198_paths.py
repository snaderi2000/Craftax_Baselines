"""Causal, matched-donor path pruning for the frozen critic's neuron 198.

Prunes messages, not weights: surrounding state-dependent computation stays at
the matched clean/donor state. Discovery episodes are divided into fit/dev;
held-out and previously used external episodes never choose circuit masks.
"""
import argparse,csv,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concept_mapping.trace_diamond_input_to_value import ROOT,read,write,branches,forward,digest

BANK=ROOT/'spatially_diverse_bank'
SOURCE=ROOT/'spatially_diverse_input_circuit'
OUT=ROOT/'neuron198_causal_pruning'
Q=np.array([1.,-1.,1.,-1.])
FACTORS={'diamond':([0,1],[3,2]),'pickaxe':([0,3],[1,2])}


def measure(zr,zd,clean,donor):
    restored=np.tanh(zr);removed=np.tanh(zd)
    contrast=Q if len(clean)==4 else np.array([1.,-1.])
    target=contrast@(clean-donor)
    recovery=contrast@(restored-donor)
    residual=contrast@(removed-donor)
    scale=max(float(np.abs(target).mean()),1e-8)
    return dict(activation_mae=float(np.abs(restored-clean).mean()),
        recovery_error=float(np.abs(recovery-target).mean()/scale),
        recovery_pct=float(100*recovery.mean()/target.mean()),
        residual_error=float(np.abs(residual).mean()/scale),
        residual_pct=float(100*residual.mean()/target.mean()),
        hA_restored=float(restored[0].mean()),hA_removed=float(removed[0].mean()))


def passes(m):
    return m['activation_mae']<=.05 and m['recovery_error']<=.10 and 90<=m['recovery_pct']<=110 and m['residual_error']<=.15 and abs(m['residual_pct'])<=10


def cost(m):return m['activation_mae']/.05+m['recovery_error']/.10+m['residual_error']/.15


class PathModel:
    def __init__(self,full,W,b,indices,factor,parents=None,mask=None):
        h,z,v=full;r,d=([0,1,2,3],[2,2,2,2]) if factor=='joint' else FACTORS[factor];self.indices=np.asarray(indices);self.factor=factor
        self.clean=h[3][r][:,indices,198].astype(np.float64)
        self.donor=h[3][d][:,indices,198].astype(np.float64)
        self.zclean=z[2][r][:,indices,198].astype(np.float64)
        self.zdonor=z[2][d][:,indices,198].astype(np.float64)
        self.w=W[2][:,198].astype(np.float64)
        self.parents=None if parents is None else np.asarray(parents)
        if parents is None:
            self.messages=(h[2][r][:,indices]-h[2][d][:,indices]).astype(np.float64)*self.w
        else:
            self.delta1=(h[1][r][:,indices]-h[1][d][:,indices]).astype(np.float64)
            self.w2=W[1][:,self.parents].astype(np.float64)
            self.h2r=h[2][r][:,indices][:,:,self.parents].astype(np.float64)
            self.h2d=h[2][d][:,indices][:,:,self.parents].astype(np.float64)
            self.z2r=z[1][r][:,indices][:,:,self.parents].astype(np.float64)
            self.z2d=z[1][d][:,indices][:,:,self.parents].astype(np.float64)
        self.mask=np.ones(512,bool) if mask is None and parents is None else np.ones((len(parents),512),bool) if mask is None else mask.copy()
        self.reset()

    def reset(self):
        if self.parents is None:
            message=self.messages@self.mask.astype(float)
            self.zr=self.zdonor+message;self.zn=self.zclean-message
        else:
            delta=self.delta1@(self.w2*self.mask.T)
            self.a2=self.z2d+delta;self.n2=self.z2r-delta
            self.zr=self.zdonor+((np.tanh(self.a2)-self.h2d)*self.w[self.parents]).sum(-1)
            self.zn=self.zclean+((np.tanh(self.n2)-self.h2r)*self.w[self.parents]).sum(-1)

    def metric(self):return measure(self.zr,self.zn,self.clean,self.donor)

    def propose(self,group,parent=None,subset=None):
        ix=slice(None) if subset is None else subset
        if self.parents is None:
            delta=self.messages[:,ix][:,:,group].sum(-1)
            return self.zr[:,ix]-delta,self.zn[:,ix]+delta,None,None
        delta=self.delta1[:,ix][:,:,group]@self.w2[group,parent]
        a2=self.a2[:,ix,parent]-delta;n2=self.n2[:,ix,parent]+delta
        w=self.w[self.parents[parent]]
        zr=self.zr[:,ix]+w*(np.tanh(a2)-np.tanh(self.a2[:,ix,parent]))
        zn=self.zn[:,ix]+w*(np.tanh(n2)-np.tanh(self.n2[:,ix,parent]))
        return zr,zn,a2,n2

    def commit(self,group,parent=None,proposal=None):
        zr,zn,a2,n2=self.propose(group,parent) if proposal is None else proposal
        self.zr,self.zn=zr,zn
        if parent is None:self.mask[group]=False
        else:
            self.mask[parent,group]=False;self.a2[:,:,parent]=a2;self.n2[:,:,parent]=n2


def prune(fit,dev,seed,label,history):
    rng=np.random.default_rng(seed);subset=np.sort(rng.choice(fit.clean.shape[1],min(512,fit.clean.shape[1]),replace=False))
    best=fit.mask.copy();best_count=int(best.sum());accepted=0
    def record(chunk,parent):
        nonlocal best,best_count
        fm,dm=fit.metric(),dev.metric();count=int(fit.mask.sum())
        row=dict(stage=label,remaining_edges=count,chunk=chunk,parent='' if parent is None else int(fit.parents[parent]),
            **{f'fit_{k}':v for k,v in fm.items()},**{f'dev_{k}':v for k,v in dm.items()},dev_passes=passes(dm))
        history.append(row)
        if passes(dm) and count<=best_count:best=fit.mask.copy();best_count=count
    record(0,None)
    parent_order=[None] if fit.parents is None else list(range(len(fit.parents)))
    for chunk in [128,64,32,16,8,4,2,1]:
        for parent in parent_order:
            active=np.flatnonzero(fit.mask if parent is None else fit.mask[parent])
            ranked=[]
            for edge in active:
                zr,zn,_,_=fit.propose([int(edge)],parent,subset)
                ranked.append((cost(measure(zr,zn,fit.clean[:,subset],fit.donor[:,subset])),int(edge)))
            ordered=[edge for _,edge in sorted(ranked)]
            for start in range(0,len(ordered),chunk):
                group=ordered[start:start+chunk]
                proposal=fit.propose(group,parent)
                fm=measure(proposal[0],proposal[1],fit.clean,fit.donor)
                if not passes(fm):continue
                fit.commit(group,parent,proposal);dev.commit(group,parent);accepted+=1;record(chunk,parent)
        print(f'{label}: chunk={chunk}, fit edges={fit.mask.sum()}, selected dev-valid edges={best_count}',flush=True)
    return best,dict(accepted_groups=accepted,fit_final_edges=int(fit.mask.sum()),selected_edges=best_count)


def cluster_bootstrap(data,ids,draws,seed):
    lookup={};inverse=[]
    for eid in ids:
        if eid not in lookup:lookup[eid]=len(lookup)
        inverse.append(lookup[eid])
    sums=np.zeros((len(lookup),data.shape[1]));np.add.at(sums,inverse,data)
    counts=np.bincount(inverse);rng=np.random.default_rng(seed)
    weights=rng.multinomial(len(lookup),np.full(len(lookup),1/len(lookup)),size=draws).astype(float)
    return (weights@sums)/(weights@counts)[:,None]


def evaluate(model,full,indices,factor,rows,draws,seed):
    h,z,v=full;r,d=([0,1,2,3],[2,2,2,2]) if factor=='joint' else FACTORS[factor];clean=model.clean;donor=model.donor
    restored=np.tanh(model.zr);removed=np.tanh(model.zn)
    originalI=Q@h[3][:,indices,198].astype(float)
    fullI=Q@v[:,indices].astype(float)
    contrast=Q if len(clean)==4 else np.array([1.,-1.])
    restI=contrast@(restored-donor)
    remI=contrast@(removed-donor)
    # Only the target neuron is modified at the final layer. Other final-hidden
    # neurons are the full actual clean/donor activations for their branches.
    wout=float(EVAL_WEIGHTS[3][198,0])
    value_after=v[:,indices].astype(float).copy()
    target_removed=h[3][:,indices,198].astype(float).copy();target_restored=target_removed.copy()
    for j,receiver in enumerate(r):
        value_after[receiver]+=wout*(removed[j]-clean[j])
        target_removed[receiver]=removed[j];target_restored[receiver]=restored[j]
    afterI=Q@value_after
    ids=[(rows[int(i)]['seed'],rows[int(i)]['episode_id']) for i in indices]
    data=np.column_stack([originalI,restI,remI,fullI,fullI-afterI,np.abs(restI-originalI),np.abs(remI),np.abs(restored-clean).mean(0)])
    boot=cluster_bootstrap(data,ids,draws,seed)
    low,high=np.quantile(boot,[.025,.975],axis=0)
    m=model.metric();m.update(n_states=len(indices),n_episodes=len(set(ids)),original_h_interaction=float(originalI.mean()),
        restored_h_interaction=float(restI.mean()),removed_h_interaction=float(remI.mean()),
        recovery_ci_low=float(np.quantile(100*boot[:,1]/boot[:,0],.025)),recovery_ci_high=float(np.quantile(100*boot[:,1]/boot[:,0],.975)),
        residual_ci_low=float(np.quantile(100*boot[:,2]/boot[:,0],.025)),residual_ci_high=float(np.quantile(100*boot[:,2]/boot[:,0],.975)),
        original_full_value_interaction=float(fullI.mean()),whole_value_interaction_removed_pct=float(100*(fullI-afterI).mean()/fullI.mean()),
        isolated_path_value_interaction=float(wout*restI.mean()),
        mean_value_change_A=float((value_after[0]-v[0,indices]).mean()),passes=passes(m))
    records=[]
    for pos,index in enumerate(indices):
        row=dict(state_index=int(index),seed=rows[int(index)]['seed'],episode_id=rows[int(index)]['episode_id'],stratum=rows[int(index)].get('stratum','external_random'),
            original_h_interaction=float(originalI[pos]),restored_h_interaction=float(restI[pos]),removed_h_interaction=float(remI[pos]),
            original_full_value_interaction=float(fullI[pos]),removed_full_value_interaction=float(afterI[pos]))
        for c,key in enumerate('ABCD'):
            row[f'h198_{key}']=float(h[3][c,index,198]);row[f'h198_{key}_removed']=float(target_removed[c,pos]);row[f'h198_{key}_restored']=float(target_restored[c,pos])
            row[f'V_{key}']=float(v[c,index]);row[f'V_{key}_removed']=float(value_after[c,pos])
        records.append(row)
    return m,records


EVAL_WEIGHTS=None


def main():
    global EVAL_WEIGHTS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--seed',type=int,default=50198)
    parser.add_argument('--bootstrap-draws',type=int,default=10000)
    parser.add_argument('--random-draws',type=int,default=100)
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True);began=time.monotonic()
    with np.load(BANK/'base_states.npz') as d:x=branches(d['observations'])
    rows=read(BANK/'episode_split.csv');all_ids={(r['seed'],r['episode_id']) for r in rows}
    train_ids=sorted({(r['seed'],r['episode_id']) for r in rows if r['split']=='discovery'})
    rng=np.random.default_rng(args.seed);order=rng.permutation(len(train_ids));nfit=int(.8*len(order))
    fit_ids={train_ids[i] for i in order[:nfit]};dev_ids={train_ids[i] for i in order[nfit:]}
    fit=np.array([i for i,r in enumerate(rows) if (r['seed'],r['episode_id']) in fit_ids]);dev=np.array([i for i,r in enumerate(rows) if (r['seed'],r['episode_id']) in dev_ids])
    test=np.array([i for i,r in enumerate(rows) if r['split']=='heldout'])
    assert not fit_ids&dev_ids and not (fit_ids|dev_ids)&{(rows[i]['seed'],rows[i]['episode_id']) for i in test}
    with np.load(SOURCE/'frozen_critic_weights.npz') as d:W=[d[f'W{j}'] for j in range(4)];b=[d[f'b{j}'] for j in range(4)]
    EVAL_WEIGHTS=W;weight_hash=digest(SOURCE/'frozen_critic_weights.npz')
    full=forward(x,W,b)
    with np.load(SOURCE/'fresh_base_states.npz') as d:external_x=branches(d['observations'])
    external_rows=read(SOURCE/'fresh_state_selection.csv');external_full=forward(external_x,W,b)
    assert not all_ids&{(r['seed'],r['episode_id']) for r in external_rows}
    protocol=dict(checkpoint=9999941632,run='wandb/run-20260630_214658-m0mw4end',target='L3 neuron 198',
        fit_episodes=len(fit_ids),dev_episodes=len(dev_ids),test_episodes=len({(rows[i]['seed'],rows[i]['episode_id']) for i in test}),
        fit_states=len(fit),dev_states=len(dev),test_states=len(test),external_states=len(external_rows),
        circuit='Factor-specific input -> selected L1 -> selected L2 -> L3-198 -> value paths. All surrounding messages retain actual matched clean/donor activity.',
        corruption='Diamond A vs D and B vs C; pickaxe A vs B and D vs C. All visible diamonds become stone, or only iron-pickaxe inventory count becomes zero.',
        selection='Actual nonlinear matched-donor intervention losses on fit episodes. Smallest accepted snapshot satisfying fixed criteria on dev episodes. Test/external never select masks.',
        criteria=dict(activation_mae_max=.05,recovery_normalized_mae_max=.10,recovery_mean_pct_range=[90,110],necessity_normalized_residual_max=.15,necessity_mean_residual_pct_max=10),
        ranking='At each pruning granularity rank individual deletions by actual intervention loss on up to 512 fixed fit states; accept deletion groups only after testing all fit states.',
        upstream_input_support='Union of actually changed input coordinates on discovery episodes. Original clean L1 signals are reconstructed from donor preactivations plus those input messages. Supporting context remains supplied by the original model.',
        no_new_training_or_rollouts=True,seed=args.seed,bootstrap_draws=args.bootstrap_draws,random_draws=args.random_draws,
        limitations='Conditional path circuit, not a standalone replacement network. Same previously used held-out/external episodes. One trained model, no actor edits.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    split_rows=[];fit_set=set(fit);dev_set=set(dev)
    for i,r in enumerate(rows):split_rows.append(dict(**r,pruning_split='fit' if i in fit_set else 'dev' if i in dev_set else 'heldout'))
    write(out/'pruning_episode_split.csv',split_rows)
    results=[];edge_rows=[];history=[];circuits={};details={};random_rows=[]
    for factor in FACTORS:
        print(f'Pruning {factor} paths; fit={len(fit)}, dev={len(dev)} states.',flush=True)
        fit_model=PathModel(full,W,b,fit,factor);dev_model=PathModel(full,W,b,dev,factor)
        np.testing.assert_allclose(fit_model.zr,fit_model.zclean,atol=1e-5)
        np.testing.assert_allclose(fit_model.zn,fit_model.zdonor,atol=1e-5)
        mask,node_run=prune(fit_model,dev_model,args.seed,factor+'_L2_to_198',history)
        parents=np.flatnonzero(mask)
        print(f'{factor}: selected L2 parents {parents.tolist()}',flush=True)
        fit_model=PathModel(full,W,b,fit,factor,parents);dev_model=PathModel(full,W,b,dev,factor,parents)
        assert passes(dev_model.metric()),'The selected node circuit should pass dev before upstream pruning.'
        upstream_mask,edge_run=prune(fit_model,dev_model,args.seed+1,factor+'_L1_to_L2',history)
        live=upstream_mask.any(1);parents=parents[live];upstream_mask=upstream_mask[live]
        active_l1=np.flatnonzero(upstream_mask.any(0));r,d=FACTORS[factor]
        source_inputs=np.flatnonzero(np.any(x[r][:,np.r_[fit,dev]]!=x[d][:,np.r_[fit,dev]],axis=(0,1)))
        circuit=dict(factor=factor,L2_parents=parents.tolist(),L1_parents=active_l1.tolist(),input_coordinates=source_inputs.tolist(),
            input_edges=len(source_inputs)*len(active_l1),L1_to_L2_edges=int(upstream_mask.sum()),L2_to_198_edges=len(parents),
            output_edges=1,node_pruning=node_run,edge_pruning=edge_run,
            scope='Only selected paths carry the factor change; every other message remains at its matched clean/donor activation.')
        circuits[factor]=circuit
        np.savez_compressed(out/f'{factor}_path_mask.npz',L2_parents=parents,L1_to_L2_mask=upstream_mask,input_coordinates=source_inputs,L1_parents=active_l1)
        (out/'selection.json').write_text(json.dumps(circuits,indent=2)+'\n')
        for src in source_inputs:
            for dest in active_l1:edge_rows.append(dict(factor=factor,layer=1,source=int(src),target=int(dest),weight=float(W[0][src,dest])))
        for j,src in np.argwhere(upstream_mask):edge_rows.append(dict(factor=factor,layer=2,source=int(src),target=int(parents[j]),weight=float(W[1][src,parents[j]])))
        for src in parents:edge_rows.append(dict(factor=factor,layer=3,source=int(src),target=198,weight=float(W[2][src,198])))
        edge_rows.append(dict(factor=factor,layer=4,source=198,target=0,weight=float(W[3][198,0])))
        # Freeze all masks before looking at held-out effects.
        for population,idx,ff,rr in [('fit',fit,full,rows),('dev',dev,full,rows),('heldout',test,full,rows),('external',np.arange(len(external_rows)),external_full,external_rows)]:
            model=PathModel(ff,W,b,idx,factor,parents,upstream_mask)
            report,per_state=evaluate(model,ff,idx,factor,rr,args.bootstrap_draws,args.seed)
            report.update(factor=factor,population=population);results.append(report)
            details[(factor,population)]=per_state
            write(out/f'{factor}_{population}_per_state.csv',per_state)
            print(f'{factor} {population}: recovered {report["recovery_pct"]:.2f}%, normalized error {report["recovery_error"]:.3f}, residual {report["residual_pct"]:.2f}%',flush=True)
        # Random structural controls, same layerwise incoming-edge counts.
        rng=np.random.default_rng(args.seed+77)
        for draw in range(args.random_draws):
            rp=rng.choice(512,len(parents),replace=False)
            rm=np.zeros_like(upstream_mask)
            for j in range(len(rp)):rm[j,rng.choice(512,int(upstream_mask[j].sum()),replace=False)]=True
            model=PathModel(full,W,b,test,factor,rp,rm);m=model.metric()
            random_rows.append(dict(factor=factor,draw=draw,**m))
        # Literal input support replay: the selected L1 signals can be reached
        # from donor states using precisely the factor's changed input slots.
        idx=test[:64]
        actual=full[0][1][r][:,idx][:,:,active_l1]
        donor_z=full[1][0][d][:,idx][:,:,active_l1]
        dx=x[r][:,idx][:,:,source_inputs]-x[d][:,idx][:,:,source_inputs]
        rebuilt=np.tanh(donor_z+dx@W[0][np.ix_(source_inputs,active_l1)])
        np.testing.assert_allclose(rebuilt,actual,atol=1e-5,rtol=1e-5)
    write(out/'pruning_history.csv',history);write(out/'selected_edges.csv',edge_rows);write(out/'summary.csv',results);write(out/'random_controls.csv',random_rows)
    joint=joint_union_test(out,W,b,full,x,rows,test,external_full,external_x,external_rows,args.bootstrap_draws,args.seed)
    plot(out,results,history,circuits,random_rows)
    render_examples(out,x,rows,test,details)
    summary=dict(protocol=protocol,circuits=circuits,results=results,joint_union=joint,
        validation=dict(weights_unchanged=digest(SOURCE/'frozen_critic_weights.npz')==weight_hash,disjoint_fit_dev_heldout=True,
            original_full_mask_restores_clean=True,original_full_mask_removes_to_donor=True,input_to_L1_path_replay=True),
        elapsed_seconds=time.monotonic()-began,
        caution='Circuit metrics concern the 198 contribution, not the whole value function. Source/context computations outside the routed paths are preserved, not explained. The pruning is greedy, so the result is not globally minimal.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');write_report(out,summary,random_rows)
    print(json.dumps(summary,indent=2),flush=True)


def joint_union_test(out,W,b,full,x,rows,test,external_full,external_x,external_rows,draws,seed):
    """Test composition of the two selected paths from the neither baseline.

    No further pruning: this test is not used to change either selected mask.
    It detects additional paths introduced by combining the two edge sets.
    """
    m1=np.zeros((1345,512),bool);m2=np.zeros((512,512),bool);selected=[]
    for factor in FACTORS:
        with np.load(out/f'{factor}_path_mask.npz') as d:
            inputs=d['input_coordinates'];l1=d['L1_parents'];parents=d['L2_parents'];mask=d['L1_to_L2_mask']
        m1[np.ix_(inputs,l1)]=True
        for j,parent in enumerate(parents):m2[mask[j],parent]=True
        selected.extend(parents.tolist())
    l1=np.flatnonzero(m1.any(0));inputs=np.flatnonzero(m1.any(1));parents=np.array(sorted(set(selected)))
    w1=W[0][np.ix_(inputs,l1)]*m1[np.ix_(inputs,l1)]
    w2=W[1][np.ix_(l1,parents)]*m2[np.ix_(l1,parents)]
    w3=W[2][parents,198]
    reports=[];records=[]
    for pop,ff,xx,rr,idx in [('heldout',full,x,rows,test),('external',external_full,external_x,external_rows,np.arange(len(external_rows)))]:
        h,z,v=ff
        dx=xx[:,idx][:,:,inputs]-xx[2,idx][:,inputs][None]
        # Retain all actual reference messages from the neither branch; route
        # just the union of selected original edges forward from input changes.
        h1=np.tanh(z[0][2,idx][:,l1][None]+dx@w1)
        h2=np.tanh(z[1][2,idx][:,parents][None]+(h1-h[1][2,idx][:,l1][None])@w2)
        restored=np.tanh(z[2][2,idx,198][None]+(h2-h[2][2,idx][:,parents][None])@w3)
        # Reverse the same input-to-target paths inside each factual branch.
        h1off=np.tanh(z[0][:,idx][:,:,l1]-dx@w1)
        h2off=np.tanh(z[1][:,idx][:,:,parents]+(h1off-h[1][:,idx][:,:,l1])@w2)
        removed=np.tanh(z[2][:,idx,198]+(h2off-h[2][:,idx][:,:,parents])@w3)
        original=h[3][:,idx,198].astype(float);target=Q@original;recover=Q@restored;residual=Q@removed
        ids=[(rr[int(i)]['seed'],rr[int(i)]['episode_id']) for i in idx]
        boot=cluster_bootstrap(np.column_stack([target,recover,residual]),ids,draws,seed)
        scale=np.abs(target).mean()
        gate=lambda a:(a[0]>0)&(a[1]<0)&(a[2]<0)&(a[3]<0)
        report=dict(population=pop,n_states=len(idx),n_episodes=len(set(ids)),
            original_interaction=float(target.mean()),recovered_interaction=float(recover.mean()),residual_interaction=float(residual.mean()),
            recovery_pct=float(100*recover.mean()/target.mean()),residual_pct=float(100*residual.mean()/target.mean()),
            recovery_error=float(np.abs(recover-target).mean()/scale),residual_error=float(np.abs(residual).mean()/scale),
            activation_mae=float(np.abs(restored-original).mean()),original_AND_pattern_pct=float(100*gate(original).mean()),
            recovered_AND_pattern_pct=float(100*gate(restored).mean()),
            recovery_ci_low=float(np.quantile(100*boot[:,1]/boot[:,0],.025)),recovery_ci_high=float(np.quantile(100*boot[:,1]/boot[:,0],.975)),
            mean_original_h={k:float(original[c].mean()) for c,k in enumerate('ABCD')},
            mean_restored_h={k:float(restored[c].mean()) for c,k in enumerate('ABCD')},
            mean_removed_h={k:float(removed[c].mean()) for c,k in enumerate('ABCD')})
        reports.append(report)
        for j,index in enumerate(idx):
            record=dict(population=pop,state_index=int(index),seed=rr[int(index)]['seed'],episode_id=rr[int(index)]['episode_id'])
            for c,key in enumerate('ABCD'):
                record[f'h_{key}']=float(original[c,j]);record[f'h_{key}_restored']=float(restored[c,j]);record[f'h_{key}_removed']=float(removed[c,j])
            records.append(record)
    write(out/'joint_union_per_state.csv',records)
    result=dict(selected_L2_parents=parents.tolist(),selected_L1_neurons=len(l1),input_to_L1_edges=int(m1.sum()),
        L1_to_L2_edges=int(m2.sum()),L2_to_198_edges=len(parents),output_edges=1,reports=reports,
        scope='Union composition test with neither-state background. No masks reselected after this test.')
    (out/'joint_union.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'joint_union_masks.npz',input_mask=m1,L1_to_L2_mask=m2,L2_parents=parents)
    core_diagram(out,W,full,test,parents,result)
    return result


def core_diagram(out,W,full,test,parents,joint):
    from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
    plt.rcParams.update({'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','font.size':10,'pdf.fonttype':42})
    s=json.loads((out/'selection.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(15,5.2),gridspec_kw={'width_ratios':[2.3,1]},constrained_layout=True);ax=axes[0];ax.set(xlim=(0,14.7),ylim=(-.7,7.7));ax.axis('off')
    def box(x,y,w,h,label,color='#eff6ff',size=8):
        ax.add_patch(FancyBboxPatch((x,y-h/2),w,h,boxstyle='round,pad=.06',facecolor=color,edgecolor='#475569'))
        ax.text(x+w/2,y,label,ha='center',va='center',fontweight='bold',fontsize=size)
    def arrow(a,b,color='#64748b',lw=1.2):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,color=color,lw=lw))
    box(.1,5.7,2,1.1,'Diamond / stone\n124 input slots')
    box(.1,1.4,2,1.1,'Iron pickaxe\nSlot 1331',color='#f3e8ff')
    box(2.8,5.7,2.3,1.1,f'{len(s["diamond"]["L1_parents"])} L1 units\n256 edges to L2')
    box(2.8,1.4,2.3,1.1,f'{len(s["pickaxe"]["L1_parents"])} L1 units\n262 edges to L2',color='#f3e8ff')
    arrow((2.15,5.7),(2.7,5.7),'#2563eb');arrow((2.15,1.4),(2.7,1.4),'#9333ea')
    ys=dict(zip([159,334,461,195,236,256],[6.7,5.55,4.4,3.25,2.1,.95]))
    for parent in [159,334,461,195,236,256]:
        y=ys[parent];box(6,y,1.2,.65,str(parent),color='#fef3c7' if parent==159 else '#eff6ff')
        if parent in s['diamond']['L2_parents']:arrow((5.18,5.7),(5.9,y),'#2563eb')
        if parent in s['pickaxe']['L2_parents']:arrow((5.18,1.4),(5.9,y),'#9333ea')
        weight=float(W[2][parent,198]);arrow((7.28,y),(9.35,3.8),'#dc2626' if weight<0 else '#2563eb',1+abs(weight))
        ax.text(8.0,.6*y+1.52,f'{weight:+.3f}',fontweight='bold',fontsize=8,bbox=dict(facecolor='white',edgecolor='none',pad=.5))
    box(9.4,3.8,2.2,1.0,'Neuron 198\ntanh',color='#fef3c7',size=10)
    box(12.4,3.8,2.1,.9,'Value',color='#fef3c7',size=9)
    arrow((11.68,3.8),(12.3,3.8));ax.text(11.6,4.5,f'{float(W[3][198,0]):+.4f}',fontsize=8,fontweight='bold')
    ax.text(.1,-.32,'Surrounding messages use matched, state-dependent context.\nGrouped arrows summarize paths; the two L1 groups overlap.',fontsize=8,fontweight='bold')
    ax.set_title('Selected conditional pathways into neuron 198')
    hh=full[0][3][:,test,198]
    rr=joint['reports'][0]
    matrix=np.array([hh.mean(1),[rr['mean_restored_h'][k] for k in 'ABCD'],[rr['mean_removed_h'][k] for k in 'ABCD']])[:,[0,1,3,2]]
    aa=axes[1];pic=aa.imshow(matrix,vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto')
    aa.set_xticks(range(4),['Both','Diamond only','Pickaxe only','Neither'],rotation=20,ha='right')
    aa.set_yticks(range(3),['Original','Joint restored','Joint removed'])
    aa.set_title('Composition test on held-out episodes')
    for i in range(3):
        for j in range(4):aa.text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',fontweight='bold',color='white' if abs(matrix[i,j])>.6 else 'black')
    fig.colorbar(pic,ax=aa,shrink=.8,label='Mean activation')
    save(fig,out,'circuit_diagram')


def save(fig,out,name):
    fig.savefig(out/f'{name}.pdf');fig.savefig(out/f'{name}.png',dpi=240);plt.close(fig)


def plot(out,results,history,circuits,random):
    plt.rcParams.update({'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,factor in zip(axes,FACTORS):
        rr=[r for r in results if r['factor']==factor and r['population'] in ['heldout','external']]
        x=np.arange(len(rr))
        ax.bar(x-.18,[100]*len(rr),width=.18,color='#6b7280',label='Original target interaction')
        ax.bar(x,[r['recovery_pct'] for r in rr],width=.18,color='#2563eb',label='Paths restored')
        ax.bar(x+.18,[r['residual_pct'] for r in rr],width=.18,color='#dc2626',label='Paths removed')
        ax.set_xticks(x,[r['population'] for r in rr]);ax.set_title(factor.capitalize()+' paths');ax.set_ylabel('Original neuron-198 interaction (%)');ax.grid(axis='y',alpha=.2)
    axes[0].legend(frameon=False,prop={'size':8,'weight':'bold'});save(fig,out,'necessity_and_recovery')
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,factor in zip(axes,FACTORS):
        rr=[r for r in history if r['stage']==factor+'_L1_to_L2']
        ax.plot([r['remaining_edges'] for r in rr],[r['fit_recovery_error'] for r in rr],label='Fit',color='#6b7280')
        ax.plot([r['remaining_edges'] for r in rr],[r['dev_recovery_error'] for r in rr],label='Dev',color='#2563eb')
        ax.axhline(.1,color='#dc2626',ls='--',lw=1);ax.set(xscale='log',xlabel='Remaining L1 → L2 messages',ylabel='Normalized interaction recovery error',title=factor.capitalize());ax.grid(alpha=.2)
    axes[0].legend(frameon=False,prop={'weight':'bold','size':8});save(fig,out,'causal_pruning_frontier')
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,factor in zip(axes,FACTORS):
        rr=[r for r in random if r['factor']==factor]
        ax.scatter([r['recovery_error'] for r in rr],[r['residual_error'] for r in rr],s=12,c='#9ca3af',alpha=.5,label='Random matched size')
        r=next(r for r in results if r['factor']==factor and r['population']=='heldout')
        ax.scatter(r['recovery_error'],r['residual_error'],s=60,c='#2563eb',label='Selected paths')
        ax.set(xlabel='Recovery error',ylabel='Residual after path removal',title=factor.capitalize());ax.grid(alpha=.2)
    axes[0].legend(frameon=False,prop={'weight':'bold','size':8});save(fig,out,'random_controls')


def render_examples(out,x,rows,test,details):
    from PIL import Image,ImageDraw,ImageFont,JpegImagePlugin,PdfImagePlugin
    from craftax.craftax_classic.constants import load_all_textures
    from concept_mapping.visualize_symbolic_trajectory import render_local_view
    strata=['adjacent_facing','adjacent_turn_required','distance_2','distance_3_4','distance_5_7']
    rng=np.random.default_rng(50199);choices=[]
    for group in strata:choices.append(int(rng.choice([i for i in test if rows[i]['stratum']==group])))
    textures=load_all_textures(7);pt=load_all_textures(28);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',17)
    selections=[]
    for factor in FACTORS:
        dest=out/f'{factor}_examples';dest.mkdir(exist_ok=True);sheets=[]
        lookup={r['state_index']:r for r in details[(factor,'heldout')]}
        for number,index in enumerate(choices,1):
            result=lookup[index];panels=[]
            for c,key in zip([0,1,3,2],['A','B','D','C']):
                view=render_local_view(x[c,index],block_size=7,include_inventory=True,display_scale=4,textures=textures,panel_textures=pt)
                panel=Image.new('RGB',(view.width,view.height+114),'white');panel.paste(view,(0,0));draw=ImageDraw.Draw(panel)
                labels=[f'value: {result[f"V_{key}"]:.4f} → {result[f"V_{key}_removed"]:.4f}',
                    f'h198 original: {result[f"h198_{key}"]:+.3f}',f'h198 removed: {result[f"h198_{key}_removed"]:+.3f}',
                    f'h198 restored: {result[f"h198_{key}_restored"]:+.3f}']
                for line,label in enumerate(labels):draw.text((3,view.height+5+line*25),label,font=font,fill='black' if line==0 else '#1d4ed8')
                panels.append(panel)
            w,h=panels[0].size;sheet=Image.new('RGB',(2*w+18,2*h+18),'white')
            for j,panel in enumerate(panels):sheet.paste(panel,((j%2)*(w+18),(j//2)*(h+18)))
            filename=f'example_{number:02d}.png';sheet.save(dest/filename);sheets.append(sheet)
            selections.append(dict(factor=factor,file=filename,state_index=index,stratum=rows[index]['stratum'],seed=rows[index]['seed'],episode_id=rows[index]['episode_id']))
        sheets[0].save(dest/'all_examples.pdf',save_all=True,append_images=sheets[1:])
    write(out/'example_selection.csv',selections)


def write_report(out,s,random):
    p=s['protocol'];lines=['# Causal path pruning into neuron 198','',
        'Frozen regular PPO critic, standard rewards, training seed 50, checkpoint 9,999,941,632. No training or new rollouts.','',
        f'Fit: {p["fit_states"]} states from {p["fit_episodes"]} discovery episodes. Dev: {p["dev_states"]} states from {p["dev_episodes"]} separate discovery episodes. '
        f'Held-out: {p["test_states"]} states from {p["test_episodes"]} episodes. External: 61 previously examined episode-disjoint states.','',
        'The surrounding critic remains active: omitted messages use the actual matched clean/corrupted activations for each state, never fixed means. '
        'The candidate paths route the factor change from the observation through selected L1-to-L2 messages, selected L2-to-198 messages, and the existing scalar readout. '
        'This identifies a conditional path circuit, not an autonomous replacement network.','',
        '## Fixed criteria and selection','',
        'Recovery normalized interaction error ≤0.10; restored activation MAE ≤0.05; mean interaction recovery 90–110%; '
        'necessity normalized residual ≤0.15 and absolute mean residual ≤10%. Group deletions are accepted only if all fit-state criteria pass. '
        'Choose the smallest recorded mask that also passes all dev criteria. No held-out or external result selects masks.','',
        '## Selected circuits','',
        '| Factor | L2 parents | L1 neurons | L1→L2 messages | Input→L1 messages |',
        '| --- | --- | ---: | ---: | ---: |']
    for f,c in s['circuits'].items():lines.append(f'| {f} | {c["L2_parents"]} | {len(c["L1_parents"])} | {c["L1_to_L2_edges"]} | {c["input_edges"]} |')
    lines.extend(['','Input edges preserve all observed changed-input coordinates into each selected L1 neuron. They were not individually greedily minimized. '
        'Their counts include positional copies. Other state-dependent input/hidden computation is required as background and is not included in these path counts.','',
        '## Evaluation','',
        '| Factor | Population | States | Recovered target interaction | Residual after removal | Recovery normalized error | Restored activation MAE | All criteria pass |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |'])
    for r in s['results']:lines.append(f'| {r["factor"]} | {r["population"]} | {r["n_states"]} | {r["recovery_pct"]:.2f}% | {r["residual_pct"]:.2f}% | {r["recovery_error"]:.3f} | {r["activation_mae"]:.3f} | {r["passes"]} |')
    lines.extend(['','Percentages are relative to the original activation interaction of neuron 198, not the full critic interaction. '
        'Necessity modifies only the target neuron at the final layer; its effect on the whole scalar value is therefore limited to its readout contribution. '
        'The 10,000-draw confidence intervals resample episodes, keeping all four branches and all states from an episode together.','',
        '## Interpretation and composition test','',
        'The diamond path selects L2 parents 159/334/461; the pickaxe path selects 159/195/236/256. '
        'Their shared parent 159 means these are overlapping conditional pathways, not independent semantic detectors. '
        'Both pathways strongly affect neuron 198 and outperform random matched-size routes. '
        'On held-out states, diamond recovery passes all fixed criteria. Pickaxe mean recovery is 98.08%, but '
        'its activation MAE is 0.0529, narrowly above the fixed 0.05 limit; it therefore does not pass every held-out criterion. '
        'Both pathways miss at least one criterion on the previously used external population.','',
        'Combining the two edge sets and restoring all four conditions from neither is a stricter composition test. '
        'The union has six L2 parents, 489 L1→L2 messages, 18,976 input messages, and six messages into 198. '
        'It recovers **86.50%** of held-out target interaction (95% episode-bootstrap CI 85.78–87.21%), below the 90% target; '
        'removing the union leaves 0.024%. Thus the factor paths are highly important but not a complete joint reconstruction. '
        'The union recovers 77.70% on external states. The combined graph was not reselected using these results. '
        '[Composition results](joint_union.json).','',
        'A separate [joint refinement](joint_refinement/README.md) directly optimizes the four-condition computation on the same fit/dev split. '
        'It is an adaptive follow-up and must not be described as an untouched replication.','',
        '## Artifacts','',
        '- [Candidate circuit diagram and composition test](circuit_diagram.pdf)',
        '- [Necessity and recovery](necessity_and_recovery.pdf)',
        '- [Causal pruning frontier](causal_pruning_frontier.pdf)',
        '- [Random controls](random_controls.pdf)',
        '- [Diamond examples](diamond_examples/all_examples.pdf), [pickaxe examples](pickaxe_examples/all_examples.pdf)',
        '- [Actual selected edges](selected_edges.csv), [selection](selection.json)',
        '- [Summary](summary.csv), [full report](summary.json), [all accepted deletions](pruning_history.csv)','',
        '## Interpretation limits','',s['caution'], '',p['limitations'],'',
        'Reproduce in jax-v100-cuda12:','', '```bash',
        'OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/mpl-causal-path python concept_mapping/prune_neuron198_paths.py',
        '```',''])
    (out/'README.md').write_text('\n'.join(lines))


if __name__=='__main__':main()

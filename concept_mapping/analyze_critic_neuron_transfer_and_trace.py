"""Cross-task single-neuron ablation and upstream edge tracing for neuron 198.

Fixed final 10B PPO critic; fixed neurons from the previous diamond analysis.
No training, new rollouts, or persistent changes to weights. Incoming-edge
rankings use discovery episodes only; patching results use held-out episodes.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from concept_mapping.plot_collectability_emergence import (
    BANKS, make_branches, read_rows, _init_network, _load_wandb_config,
    _restore_train_state, make_craftax_env_from_name,
)
from concept_mapping.probe_frozen_policy_collectability import component_arrays, fingerprint, write_csv
from concept_mapping.analyze_water_health_interaction_10k import make_counterfactuals, CONDITIONS
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path('concept_mapping/runs/diamond_critic_circuit_10b')
RUN = 'wandb/run-20260630_214658-m0mw4end'
NEURONS = (147, 198, 11, 439)
Q = np.array([1., -1., 1., -1.])  # A B C D
SIZES = (1, 4, 8, 16, 32, 64, 128, 512)


def interval(data, draws, seed):
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, draws, 100):
        ix = rng.integers(0, len(data), (min(100, draws-start), len(data)))
        estimates.append(data[ix].mean(axis=1))
    estimates = np.concatenate(estimates)
    return np.quantile(estimates, [.025,.975], axis=0).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'followup_neuron_specificity_and_trace')
    parser.add_argument('--bootstrap-draws', type=int, default=10000)
    parser.add_argument('--random-draws', type=int, default=32)
    parser.add_argument('--seed', type=int, default=50)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out/'per_state').mkdir(exist_ok=True)
    (out/'trace').mkdir(exist_ok=True)
    began = time.monotonic()
    parent = json.loads((ROOT/'protocol.json').read_text())
    checkpoint = int(parent['checkpoint'])
    split = read_rows(ROOT/'episode_split.csv')
    discovery = np.array([int(r['state_index']) for r in split if r['split']=='discovery'])
    test = np.array([int(r['state_index']) for r in split if r['split']=='heldout'])
    diamond_ids = {(int(r['seed']), int(r['episode_id'])) for r in split}
    protocol = dict(run=RUN, checkpoint=checkpoint, fixed_neurons=NEURONS, target_neuron=198,
        discovery_n=len(discovery), heldout_n=len(test), seed=args.seed,
        cross_task_selection='Diamond: original held-out episodes. Iron/plant/water: every eligible saved episode outside the entire diamond bank. No neuron reselection.',
        ablation='Replace one neuron, or all four, by the pooled diamond-discovery activation mean, using the same replacement in every condition and task.',
        trace='Capture second-hidden activations, target weighted input z198 before tanh, and h198 after tanh. Decompose z into neither baseline, two main effects, and interaction.',
        additive_reconstruction='Replace zA by zB+zD-zC, keeping B/C/D fixed; compare tanh interaction. This is a diagnostic anchored at neither, not a unique causal decomposition.',
        incoming_edge_ranking='Discovery mean absolute weighted second-hidden activation changes for diamond or pickaxe, averaged across the other factor; interaction ranking uses weighted mixed contrast.',
        edge_patching='Route donor second-hidden activations only along selected incoming edges to neuron 198. Other last-hidden neurons remain fixed.',
        primary_trace_size=16, sizes=SIZES, random_draws=args.random_draws,
        no_new_training_or_rollouts=True)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    cfg = _load_wandb_config(RUN)
    env = make_craftax_env_from_name(cfg['ENV_NAME'], auto_reset=False)
    net = _init_network(cfg, env, env.default_params)
    state, _ = _restore_train_state(RUN,cfg,net,env,env.default_params,checkpoint)
    p = state.params['params']
    initial_hash = fingerprint(component_arrays(state.params,range(8)))
    output_weights = np.asarray(p['Dense_7']['kernel']).reshape(-1)[list(NEURONS)].astype(np.float64)

    @jax.jit
    def inspect(obs):
        h = obs
        for layer in (4,5):
            h = jnp.tanh(h @ p[f'Dense_{layer}']['kernel'] + p[f'Dense_{layer}']['bias'])
        z = h @ p['Dense_6']['kernel'] + p['Dense_6']['bias']
        final = jnp.tanh(z)
        value = (final @ p['Dense_7']['kernel'] + p['Dense_7']['bias']).squeeze(-1)
        return h, z[:, jnp.array(NEURONS)], final[:, jnp.array(NEURONS)], value

    def capture(obs, keep_parents=False):
        hs, zs, ys, vs = [], [], [], []
        for start in range(0,len(obs),512):
            h,z,y,v = inspect(jnp.asarray(obs[start:start+512]))
            if keep_parents: hs.append(np.asarray(h))
            zs.append(np.asarray(z));ys.append(np.asarray(y));vs.append(np.asarray(v))
        return (np.concatenate(hs) if keep_parents else None,
                np.concatenate(zs),np.concatenate(ys),np.concatenate(vs).astype(np.float64))

    branches = make_branches('diamond',BANKS['diamond'])
    diamond_rows = read_rows(BANKS['diamond']/'per_state_results.csv')
    dh2,dz,dh,dv = [],[],[],[]
    replay_errors = {}
    for key in 'ABCD':
        h2,z,h,v = capture(branches[key],True)
        expected = np.array([float(r['V_'+key]) for r in diamond_rows])
        replay_errors['diamond_'+key] = float(np.max(np.abs(v-expected)))
        np.testing.assert_allclose(v,expected,atol=1e-5,rtol=1e-5)
        dh2.append(h2);dz.append(z);dh.append(h);dv.append(v)
    dh2,dz,dh,dv = map(np.stack,(dh2,dz,dh,dv))
    replacement = dh[:,discovery].mean(axis=(0,1)).astype(np.float64)
    tasks = dict(diamond=dict(rows=[diamond_rows[i] for i in test], branches=list('ABCD'),
        z=dz[:,test],h=dh[:,test],v=dv[:,test]))
    selection = []
    for task in ('iron','plant'):
        rows = read_rows(BANKS[task]/'per_state_results.csv')
        eligible = np.array([i for i,r in enumerate(rows) if (int(r['seed']),int(r['episode_id'])) not in diamond_ids])
        b = make_branches(task,BANKS[task])
        keys = list('ABCD') if task=='iron' else ['ripe','unripe']
        result = [capture(b[key][eligible]) for key in keys]
        z,h,v = [np.stack([r[j] for r in result]) for j in (1,2,3)]
        for j,key in enumerate(keys):
            expected = np.array([float(rows[i]['V_'+key]) for i in eligible])
            replay_errors[task+'_'+key] = float(np.max(np.abs(v[j]-expected)))
            np.testing.assert_allclose(v[j],expected,atol=1e-5,rtol=1e-5)
        selected_rows = [rows[i] for i in eligible]
        tasks[task] = dict(rows=selected_rows,branches=keys,z=z,h=h,v=v)
        selection.extend(dict(task=task,seed=r['seed'],episode_id=r['episode_id'],source_index=int(i)) for i,r in zip(eligible,selected_rows))
    water_dir = Path('concept_mapping/runs/water_health_interaction_10b_10000episodes')
    rows = read_rows(water_dir/'per_state_results.csv')
    eligible = np.array([i for i,r in enumerate(rows) if (int(r['seed']),int(r['episode_id'])) not in diamond_ids])
    with np.load(water_dir/'base_states.npz') as data:
        b = make_counterfactuals(data['observations'][eligible].astype(np.float32))
    keys = list(CONDITIONS)
    result = [capture(b[key]) for key in keys]
    z,h,v = [np.stack([r[j] for r in result]) for j in (1,2,3)]
    for j,key in enumerate(keys):
        expected = np.array([float(rows[i]['V_'+key]) for i in eligible])
        replay_errors['water_'+key] = float(np.max(np.abs(v[j]-expected)))
        np.testing.assert_allclose(v[j],expected,atol=1e-5,rtol=1e-5)
    selected_rows = [rows[i] for i in eligible]
    tasks['water'] = dict(rows=selected_rows,branches=keys,z=z,h=h,v=v)
    selection.extend(dict(task='water',seed=r['seed'],episode_id=r['episode_id'],source_index=int(i)) for i,r in zip(eligible,selected_rows))
    write_csv(out/'cross_task_episode_selection.csv',selection)

    responses, contrasts, baseline_metrics = [],[],[]
    contrast_definitions = {}
    for task,t in tasks.items():
        n = len(t['rows'])
        print(f'Cross-task {task}: {n:,} states, {len(t["branches"])} conditions',flush=True)
        if task in ('diamond','iron'):
            definitions = dict(interaction=Q,tool_with_resource=np.array([1.,-1,0,0]),
                tool_without_resource=np.array([0.,0,-1,1]),
                resource_with_tool=np.array([1.,0,0,-1]),resource_without_tool=np.array([0.,1,-1,0]))
        elif task=='plant':
            definitions = dict(ripe_advantage=np.array([1.,-1]))
        else:
            definitions = {}
            for health in (2,9):
                for drink in (0,9):
                    c = np.zeros(8)
                    c[keys.index(f'h{health}_d{drink}_water')]=1
                    c[keys.index(f'h{health}_d{drink}_no_water')]=-1
                    definitions[f'water_h{health}_d{drink}'] = c
            definitions['thirst_modulation_h2'] = definitions['water_h2_d0']-definitions['water_h2_d9']
            definitions['health_modulation_d0'] = definitions['water_h2_d0']-definitions['water_h9_d0']
            definitions['three_way_interaction'] = definitions['water_h2_d0']-definitions['water_h2_d9']-definitions['water_h9_d0']+definitions['water_h9_d9']
        contrast_definitions[task] = {key:c.tolist() for key,c in definitions.items()}
        ablated = t['v'][None] + output_weights[:,None,None]*(replacement[:,None,None]-t['h'].transpose(2,0,1))
        all_after = t['v'] + (output_weights*(replacement-t['h'])).sum(axis=2)
        for ci,key in enumerate(t['branches']):
            for j,neuron in enumerate(NEURONS):
                responses.append(dict(task=task,condition=key,neuron=neuron,n=n,
                    mean_preactivation=float(t['z'][ci,:,j].mean()),mean_activation=float(t['h'][ci,:,j].mean()),
                    activation_sd=float(t['h'][ci,:,j].std()),positive_activation_pct=float(100*(t['h'][ci,:,j]>0).mean()),
                    output_weight=float(output_weights[j]),mean_value_contribution=float(output_weights[j]*t['h'][ci,:,j].mean()),
                    mean_value=float(t['v'][ci].mean()),mean_value_after_single_ablation=float(ablated[j,ci].mean())))
        primary_key = 'interaction' if task in ('diamond','iron') else 'ripe_advantage' if task=='plant' else 'water_h2_d0'
        for metric,c in definitions.items():
            original = c @ t['v']
            contribution = np.einsum('c,cnj->nj',c,t['h'])*output_weights
            removed_all = contribution.sum(axis=1)
            assert np.allclose(c@all_after,original-removed_all,atol=1e-5)
            ci = interval(np.column_stack([original,contribution,removed_all]),args.bootstrap_draws,args.seed) if metric==primary_key else None
            baseline_metrics.append(dict(task=task,metric=metric,n=n,baseline_mean=float(original.mean()),
                after_all_four_mean=float((c@all_after).mean()),all_four_removed_fraction=float(removed_all.mean()/original.mean()) if abs(original.mean())>1e-8 else None))
            for j,neuron in enumerate(NEURONS):
                after = c @ ablated[j]
                np.testing.assert_allclose(after,original-contribution[:,j],atol=1e-5)
                contrasts.append(dict(task=task,metric=metric,neuron=neuron,n=n,
                    original_mean=float(original.mean()),mean_removed=float(contribution[:,j].mean()),
                    mean_after=float(after.mean()),removed_fraction=float(contribution[:,j].mean()/original.mean()) if abs(original.mean())>1e-8 else None,
                    removed_ci_low=ci[0][j+1] if ci else '',removed_ci_high=ci[1][j+1] if ci else '',
                    value_mae_under_ablation=float(np.abs(ablated[j]-t['v']).mean())))
        records=[]
        for i,r in enumerate(t['rows']):
            row=dict(seed=r['seed'],episode_id=r['episode_id'])
            for ci,key in enumerate(t['branches']):
                row['V_'+key]=float(t['v'][ci,i])
                row['V_'+key+'_all4_ablated']=float(all_after[ci,i])
                for j,neuron in enumerate(NEURONS):
                    row[f'h{neuron}_{key}']=float(t['h'][ci,i,j])
                    row[f'z{neuron}_{key}']=float(t['z'][ci,i,j])
                    row[f'V_{key}_ablate{neuron}']=float(ablated[j,ci,i])
            records.append(row)
        write_csv(out/'per_state'/f'{task}.csv',records)
    write_csv(out/'activation_responses.csv',responses)
    write_csv(out/'single_neuron_ablation.csv',contrasts)
    write_csv(out/'task_contrasts.csv',baseline_metrics)
    print('Single-neuron cross-task analysis complete.',flush=True)

    # Trace only the path from second-hidden neurons into final-hidden unit 198.
    weights = np.asarray(p['Dense_6']['kernel'])[:,198].astype(np.float64)
    bias = float(np.asarray(p['Dense_6']['bias'])[198])
    parent64 = dh2.astype(np.float64)
    hz = np.einsum('cni,i->cn',parent64,weights)+bias
    np.testing.assert_allclose(hz,dz[:,:,1],atol=1e-5,rtol=1e-5)
    hy = np.tanh(hz)
    np.testing.assert_allclose(hy,dh[:,:,1],atol=1e-5,rtol=1e-5)
    parts = dict(neither=hz[2],diamond_main=hz[1]-hz[2],pickaxe_main=hz[3]-hz[2],upstream_interaction=Q@hz)
    np.testing.assert_allclose(hz[0],sum(parts.values()),atol=1e-10)
    additive_z=hz.copy();additive_z[0]=hz[1]+hz[3]-hz[2]
    additive_h=np.tanh(additive_z)
    full_h_interaction=Q@hy
    h_interaction_additive=Q@additive_h
    weighted_changes=dict(
        diamond_present=(parent64[0]-parent64[3])*weights,
        diamond_absent=(parent64[1]-parent64[2])*weights,
        pickaxe_present=(parent64[0]-parent64[1])*weights,
        pickaxe_absent=(parent64[3]-parent64[2])*weights,
        upstream_interaction=np.einsum('c,cni->ni',Q,parent64)*weights)
    np.testing.assert_allclose(weighted_changes['upstream_interaction'].sum(axis=1),parts['upstream_interaction'],atol=1e-10)
    score_d=.5*(np.abs(weighted_changes['diamond_present'][discovery])+np.abs(weighted_changes['diamond_absent'][discovery])).mean(axis=0)
    score_p=.5*(np.abs(weighted_changes['pickaxe_present'][discovery])+np.abs(weighted_changes['pickaxe_absent'][discovery])).mean(axis=0)
    score_i=np.abs(weighted_changes['upstream_interaction'][discovery]).mean(axis=0)
    rankings={key:np.argsort(-score,kind='stable') for key,score in [('diamond',score_d),('pickaxe',score_p),('upstream_interaction',score_i)]}
    rank_rows=[]
    for category,ranking in rankings.items():
        for rank,neuron in enumerate(ranking,1):
            row=dict(category=category,rank=rank,parent_neuron=int(neuron),incoming_weight=float(weights[neuron]))
            for key,a in weighted_changes.items():row[key+'_heldout_mean']=float(a[test,neuron].mean())
            row['discovery_score']=float(dict(diamond=score_d,pickaxe=score_p,upstream_interaction=score_i)[category][neuron])
            rank_rows.append(row)
    write_csv(out/'trace/incoming_edge_rankings.csv',rank_rows)
    trace_records=[]
    for i in test:
        row=dict(state_index=int(i),seed=diamond_rows[i]['seed'],episode_id=diamond_rows[i]['episode_id'])
        for ci,key in enumerate('ABCD'):
            row['z_'+key]=float(hz[ci,i]);row['h_'+key]=float(hy[ci,i])
        row.update(z_interaction=float(parts['upstream_interaction'][i]),h_interaction=float(full_h_interaction[i]),
                   additive_z_A=float(additive_z[0,i]),additive_h_A=float(additive_h[0,i]),h_interaction_additive_z=float(h_interaction_additive[i]))
        trace_records.append(row)
    write_csv(out/'trace/per_state_preactivation.csv',trace_records)

    # Donor contributions are routed along selected edges to 198 only. This
    # leaves all other last-hidden activations intact (a path intervention).
    patch_rows,random_rows,patch_per_state=[],[],[]
    random=np.random.default_rng(args.seed+198)
    for category in ('diamond','pickaxe'):
        present,absent=(weighted_changes['diamond_present'],weighted_changes['diamond_absent']) if category=='diamond' else (weighted_changes['pickaxe_present'],weighted_changes['pickaxe_absent'])
        receiver_p,receiver_a=(3,2) if category=='diamond' else (1,2)
        donor_p,donor_a=(0,1) if category=='diamond' else (0,3)
        for k in SIZES:
            masks=np.zeros((1+(args.random_draws if k<512 else 1),512))
            masks[0,rankings[category][:k]]=1
            for draw in range(1,len(masks)):masks[draw,random.choice(512,k,replace=False)]=1
            new_p=np.tanh(hz[receiver_p,test][None]+masks@present[test].T)
            new_a=np.tanh(hz[receiver_a,test][None]+masks@absent[test].T)
            effect_p=new_p-hy[receiver_p,test]
            effect_a=new_a-hy[receiver_a,test]
            recovered=effect_p-effect_a
            target=full_h_interaction[test]
            if k==512:
                np.testing.assert_allclose(new_p[0],hy[donor_p,test],atol=1e-10)
                np.testing.assert_allclose(new_a[0],hy[donor_a,test],atol=1e-10)
                np.testing.assert_allclose(recovered[0],target,atol=1e-10)
            for draw in range(len(masks)):
                row=dict(category=category,k=k,method='discovery_ranked' if draw==0 else 'random',draw=draw-1,n=len(test),
                    activation_interaction_baseline=float(target.mean()),
                    mean_patch_effect_other_factor_present=float(effect_p[draw].mean()),mean_patch_effect_other_factor_absent=float(effect_a[draw].mean()),
                    recovered_activation_interaction=float(recovered[draw].mean()),recovered_fraction=float(recovered[draw].mean()/target.mean()),
                    target_activation_mae=float(.5*(np.abs(new_p[draw]-hy[donor_p,test])+np.abs(new_a[draw]-hy[donor_a,test])).mean()))
                (patch_rows if draw==0 else random_rows).append(row)
            if k==16:
                for position,i in enumerate(test):
                    patch_per_state.append(dict(category=category,state_index=int(i),
                        patched_activation_present=float(new_p[0,position]),patched_activation_absent=float(new_a[0,position]),
                        recovered_interaction=float(recovered[0,position]),original_interaction=float(target[position])))
    write_csv(out/'trace/path_patching_summary.csv',patch_rows)
    write_csv(out/'trace/random_path_controls.csv',random_rows)
    write_csv(out/'trace/top16_path_patches_per_state.csv',patch_per_state)
    primary_edges={category:rankings[category][:16] for category in ('diamond','pickaxe')}
    # Necessity: replace selected incoming signals in A by the matching missing-factor donor.
    necessity=[]
    for category,donor in [('diamond',3),('pickaxe',1)]:
        selected=primary_edges[category]
        delta=((parent64[donor,test][:,selected]-parent64[0,test][:,selected])*weights[selected]).sum(axis=1)
        new_h=np.tanh(hz[0,test]+delta)
        necessity.append(dict(factor_removed=category,parents=selected.tolist(),mean_original_h_A=float(hy[0,test].mean()),
                              mean_h_A_after_path_removal=float(new_h.mean()),positive_activation_pct_after=float(100*(new_h>0).mean())))
    trace_summary=dict(target_neuron=198,output_weight=float(output_weights[1]),incoming_bias=bias,
        heldout_n=len(test),mean_preactivation={key:float(hz[j,test].mean()) for j,key in enumerate('ABCD')},
        mean_activation={key:float(hy[j,test].mean()) for j,key in enumerate('ABCD')},
        mean_z_parts={key:float(a[test].mean()) for key,a in parts.items()},
        mean_absolute_upstream_interaction=float(np.abs(parts['upstream_interaction'][test]).mean()),
        mean_h_interaction=float(full_h_interaction[test].mean()),mean_h_interaction_additive_z=float(h_interaction_additive[test].mean()),
        additive_z_interaction_retained_fraction=float(h_interaction_additive[test].mean()/full_h_interaction[test].mean()),
        original_A_positive_pct=float(100*(hy[0,test]>0).mean()),additive_z_A_positive_pct=float(100*(additive_h[0,test]>0).mean()),
        original_AND_sign_pattern_pct=float(100*((hy[0,test]>0)&(hy[1,test]<0)&(hy[2,test]<0)&(hy[3,test]<0)).mean()),
        top16_path_patches=[r for r in patch_rows if r['k']==16],top16_path_removals=necessity,
        top16_diamond_pickaxe_overlap=[int(i) for i in sorted(set(primary_edges['diamond'])&set(primary_edges['pickaxe']))],
        validation=dict(cpu_weighted_sum_max_error=float(np.max(np.abs(hz-dz[:,:,1]))),
                        tanh_replay_max_error=float(np.max(np.abs(hy-dh[:,:,1]))),all_edge_patches_reproduce_donors=True),
        caution='An additive-z reconstruction is anchored at C and can be outside the empirical distribution. Retention is a diagnostic, not a unique allocation of computation between layers. Incoming edges may carry mixed concepts.')
    trace_summary['upstream_core'] = analyze_upstream_core(out,dh2,weights,bias,hz,hy,dv,discovery,test,rankings,args)
    (out/'trace/summary.json').write_text(json.dumps(trace_summary,indent=2)+'\n')
    np.savez_compressed(out/'trace/diamond_parent_activations.npz',second_hidden=dh2,z198=hz,h198=hy,
        incoming_weights=weights,incoming_bias=bias,discovery_indices=discovery,heldout_indices=test)
    assert fingerprint(component_arrays(state.params,range(8)))==initial_hash
    report=dict(protocol=protocol,n_by_task={key:len(t['rows']) for key,t in tasks.items()},
        discovery_replacement_activations=dict(zip(map(str,NEURONS),replacement.tolist())),
        replay_max_errors=replay_errors,parameter_sha256=initial_hash,parameters_unchanged=True,
        contrasts=baseline_metrics,trace=trace_summary,contrast_definitions=contrast_definitions,
        confidence_intervals='10,000 paired episode-bootstrap draws for each primary task contrast; conditional on fixed candidate neurons and populations.',
        limitations='One model seed/checkpoint; counterfactual states can be out of distribution. Last-layer contributions and ablations are exact linear readout effects, not proof of a complete upstream mechanism. Task populations differ. No actor behavior or simulator causality is tested.',
        elapsed_seconds=time.monotonic()-began)
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    plot_results(out,responses,contrasts,trace_summary,patch_rows,random_rows,hz[:,test],hy[:,test])
    write_report(out,report,contrasts)
    print(json.dumps(dict(n_by_task=report['n_by_task'],primary_contrasts=[r for r in contrasts if r['metric'] in ['interaction','ripe_advantage','water_h2_d0']],trace=trace_summary,elapsed=report['elapsed_seconds']),indent=2),flush=True)


def analyze_upstream_core(out,h2,weights,bias,z,h,values,discovery,test,rankings,args):
    # Exploratory simplification using only discovery rankings: top two joint
    # inputs plus the strongest pickaxe input and top two diamond inputs.
    core=sorted(set(rankings['upstream_interaction'][:2])|set(rankings['diamond'][:2])|set(rankings['pickaxe'][:1]))
    core=[int(i) for i in core]
    carriers=sorted(set([int(rankings['pickaxe'][0]),int(rankings['diamond'][1])]))
    coupled=[int(i) for i in rankings['upstream_interaction'][:2]]
    h2=h2.astype(np.float64)
    replacement=h2[:,discovery].mean(axis=(0,1))
    constant=float(bias+replacement@weights)
    def reduced(selected):
        return constant+np.einsum('cni,i->cn',h2[:,:,selected]-replacement[selected],weights[selected])
    groups={'four_parent_core':core,'diamond_and_pickaxe_carriers_only':carriers,'coupled_parents_only':coupled}
    results={};records=[];reduced_values={}
    baseline_pattern=(h[0,test]>0)&(h[1,test]<0)&(h[2,test]<0)&(h[3,test]<0)
    for name,selected in groups.items():
        zz=reduced(selected);yy=np.tanh(zz);reduced_values[name]=yy
        pattern=(yy[0,test]>0)&(yy[1,test]<0)&(yy[2,test]<0)&(yy[3,test]<0)
        results[name]=dict(parent_neurons=selected,AND_sign_pattern_pct=float(100*pattern.mean()),
            mean_z={key:float(zz[j,test].mean()) for j,key in enumerate('ABCD')},
            mean_h={key:float(yy[j,test].mean()) for j,key in enumerate('ABCD')},
            activation_mae=float(np.abs(yy[:,test]-h[:,test]).mean()),
            branch_sign_agreement_pct=float(100*((yy[:,test]>0)==(h[:,test]>0)).mean()),
            interaction_retained_fraction=float((Q@yy[:,test]).mean()/(Q@h[:,test]).mean()))
    rng=np.random.default_rng(args.seed+9198)
    random=[]
    for draw in range(args.random_draws):
        selected=rng.choice(512,len(core),replace=False)
        yy=np.tanh(reduced(selected))
        random.append(dict(draw=draw,AND_sign_pattern_pct=float(100*((yy[0,test]>0)&(yy[1,test]<0)&(yy[2,test]<0)&(yy[3,test]<0)).mean()),
            activation_mae=float(np.abs(yy[:,test]-h[:,test]).mean()),
            interaction_retained_fraction=float((Q@yy[:,test]).mean()/(Q@h[:,test]).mean())))
    write_csv(out/'trace/random_core_controls.csv',random)
    for position,i in enumerate(test):
        row=dict(state_index=int(i))
        for j,key in enumerate('ABCD'):
            row[f'h198_{key}']=float(h[j,i])
            for name,yy in reduced_values.items():row[f'h198_{key}_{name}']=float(yy[j,i])
        records.append(row)
    write_csv(out/'trace/reduced_core_per_state.csv',records)
    result=dict(selection='Exploratory union of discovery-ranked top-two upstream-interaction parents, top-two diamond parents, and top-one pickaxe parent; no held-out fitting.',
        intervention='Retain indicated incoming edges to neuron 198; replace every other parent activation by its pooled discovery mean. Other final-hidden neurons are unchanged.',
        constant_from_clamped_parents_and_bias=constant,original_AND_sign_pattern_pct=float(100*baseline_pattern.mean()),
        parent_response={str(neuron):dict(weight=float(weights[neuron]),mean_h={key:float(h2[j,test,neuron].mean()) for j,key in enumerate('ABCD')}) for neuron in core},
        reduced_models=results,random_core_median_AND_pct=float(np.median([r['AND_sign_pattern_pct'] for r in random])),
        random_core_max_AND_pct=float(max(r['AND_sign_pattern_pct'] for r in random)),
        scope='Reproduces a response of one internal neuron on the fixed diamond probe bank, not the entire critic, actor, or arbitrary observations.')
    (out/'trace/upstream_core.json').write_text(json.dumps(result,indent=2)+'\n')
    plt.rcParams.update({'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','font.size':10,'pdf.fonttype':42})
    labels=[f'L2 neuron {i}, w={weights[i]:+.2f}' for i in core]+['Original L3 neuron 198','Four-parent L3 neuron 198']
    matrix=np.array([h2[:,test,i].mean(axis=1) for i in core]+[h[:,test].mean(axis=1),reduced_values['four_parent_core'][:,test].mean(axis=1)])[:,[0,1,3,2]]
    fig,ax=plt.subplots(figsize=(8,4.8),constrained_layout=True)
    pic=ax.imshow(matrix,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
    ax.set_xticks(range(4),['Diamond + pickaxe','Diamond only','Pickaxe only','Neither'],rotation=12)
    ax.set_yticks(range(len(labels)),labels);ax.set_title('A compact upstream path to neuron 198')
    ax.axhline(len(core)-.5,color='black',lw=1)
    for i in range(len(labels)):
        for j in range(4):ax.text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',fontweight='bold',color='white' if abs(matrix[i,j])>.6 else 'black')
    fig.colorbar(pic,ax=ax,label='Held-out mean activation');save(fig,out/'trace','upstream_core_activation_patterns')
    return result


def plot_results(out,responses,contrasts,trace,patches,random_rows,z,h):
    plt.rcParams.update({'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    definitions=[('diamond',['A','B','D','C'],['Diamond + tool','Diamond only','Tool only','Neither']),
                 ('iron',['A','B','D','C'],['Iron + tool','Iron only','Tool only','Neither']),
                 ('plant',['ripe','unripe'],['Ripe','Unripe']),
                 ('water',['h2_d0_water','h2_d0_no_water'],['Water','No water'])]
    for ax,(task,keys,labels) in zip(axes.flat,definitions):
        matrix=np.array([[next(r['mean_activation'] for r in responses if r['task']==task and r['condition']==key and r['neuron']==neuron) for key in keys] for neuron in NEURONS])
        pic=ax.imshow(matrix,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
        ax.set_xticks(range(len(keys)),labels,rotation=15);ax.set_yticks(range(4),[str(n) for n in NEURONS]);ax.set_ylabel('Fixed neuron');ax.set_title(task.capitalize())
        for i in range(4):
            for j in range(len(keys)):ax.text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',fontweight='bold',color='white' if abs(matrix[i,j])>.6 else 'black')
    fig.colorbar(pic,ax=list(axes.flat),label='Mean tanh activation')
    save(fig,out,'cross_task_activation_maps')
    task_metrics=[('diamond','interaction'),('iron','interaction'),('plant','ripe_advantage'),('water','water_h2_d0')]
    matrix=np.array([[next(r['removed_fraction'] for r in contrasts if r['task']==task and r['metric']==metric and r['neuron']==neuron) for task,metric in task_metrics] for neuron in NEURONS])*100
    fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    limit=max(100,float(np.max(np.abs(matrix))))
    pic=ax.imshow(matrix,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_xticks(range(4),['Diamond × tool','Iron × tool','Ripe plant','Water (thirsty)']);ax.set_yticks(range(4),[str(n) for n in NEURONS]);ax.set_ylabel('Neuron ablated');ax.set_title('Contribution to each paired value contrast')
    for i in range(4):
        for j in range(4):ax.text(j,i,f'{matrix[i,j]:+.1f}%',ha='center',va='center',fontweight='bold',color='white' if abs(matrix[i,j])>.6*limit else 'black')
    fig.colorbar(pic,ax=ax,label='Original mean contrast removed (%)');save(fig,out,'single_neuron_specificity')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    order=[0,1,3,2];labels=['Both','Diamond only','Pickaxe only','Neither']
    for ax,array,title in zip(axes,[z,h],['Weighted input to neuron 198','After tanh']):
        ax.bar(range(4),array.mean(axis=1)[order],yerr=array.std(axis=1)[order],color=['#2563eb','#d97706','#dc2626','#0f766e'],capsize=3)
        ax.set_xticks(range(4),labels,rotation=15);ax.set_title(title);ax.set_ylabel('Mean ± SD across held-out states');ax.axhline(0,color='black',lw=.7);ax.grid(axis='y',alpha=.2)
    save(fig,out/'trace','preactivation_vs_activation')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for ax,category in zip(axes,['diamond','pickaxe']):
        selected=[r for r in patches if r['category']==category and r['k']<512]
        ks=[r['k'] for r in selected]
        vals=[[100*r['recovered_fraction'] for r in random_rows if r['category']==category and r['k']==k] for k in ks]
        ax.plot(ks,[100*r['recovered_fraction'] for r in selected],marker='o',color='#2563eb',label='Discovery-ranked edges')
        ax.plot(ks,[np.median(a) for a in vals],color='#6b7280',ls='--',label='Random median')
        ax.fill_between(ks,[min(a) for a in vals],[max(a) for a in vals],color='#6b7280',alpha=.15,label='Random min–max')
        ax.axhline(100,color='black',lw=.7,alpha=.4);ax.set(xscale='log',xlabel='Incoming edges to neuron 198',ylabel='Activation interaction recovered (%)',title=category.capitalize()+' path patch');ax.grid(alpha=.2);ax.legend(frameon=False,prop={'size':7,'weight':'bold'})
    save(fig,out/'trace','incoming_path_patching')


def save(fig,out,stem):
    fig.savefig(out/f'{stem}.pdf');fig.savefig(out/f'{stem}.png',dpi=250);plt.close(fig)


def write_report(out,report,contrasts):
    t=report['trace']
    lines=['# Fixed-neuron transfer and neuron-198 upstream trace','',
        'Same final regular-PPO 10B critic and fixed neurons 147, 198, 11, 439. No training or new rollouts. '
        'Diamond uses the original 1,045 held-out episodes; other tasks exclude all episodes in the diamond bank. '
        'Each ablation uses the original diamond discovery mean, not a task-specific replacement.','',
        '| Neuron | Diamond interaction removed | Iron interaction removed | Ripe-plant advantage removed | Water advantage removed |',
        '| --- | ---: | ---: | ---: | ---: |']
    for neuron in NEURONS:
        vals=[next(r['removed_fraction'] for r in contrasts if r['neuron']==neuron and r['task']==task and r['metric']==metric) for task,metric in [('diamond','interaction'),('iron','interaction'),('plant','ripe_advantage'),('water','water_h2_d0')]]
        lines.append('| '+str(neuron)+' | '+' | '.join(f'{100*v:.2f}%' for v in vals)+' |')
    lines.extend(['','Fractions are signed changes in the original mean contrast; negative values mean that ablating a neuron increases the contrast. '
        'These are exact last-layer readout contributions, not isolated concept labels. Task sample sizes: '+str(report['n_by_task'])+'.','',
        'Neuron 198 before tanh: z=b+sum_k W[k,198]h2[k]. A/B/C/D mean weighted inputs: '+str(t['mean_preactivation'])+'. '
        f'The mean upstream mixed contrast is {t["mean_z_parts"]["upstream_interaction"]:.4f}; after tanh, the mean activation mixed contrast is {t["mean_h_interaction"]:.4f}.','',
        'Anchoring main effects at neither, zA(additive)=zB+zD−zC removes the upstream mixed contrast. '
        f'Applying tanh to that reconstruction retains {100*t["additive_z_interaction_retained_fraction"]:.1f}% of the observed activation interaction. '
        'This tests one additive reconstruction; it is not a unique decomposition of nonlinear causal effects.','',
        'Incoming-edge importance is ranked on discovery episodes. For diamond paths, donors A→D and B→C transfer diamond information; '
        'for pickaxe paths, donors A→B and D→C transfer pickaxe information. Only selected edges into neuron 198 are patched; '
        'other last-hidden neurons are unchanged. All 512-edge endpoint patches reproduce target donor activations.','',
        '[Cross-task activation maps](cross_task_activation_maps.pdf), [single-neuron specificity](single_neuron_specificity.pdf), '
        '[preactivation versus activation](trace/preactivation_vs_activation.pdf), [incoming-edge patching](trace/incoming_path_patching.pdf).','',
        'An exploratory upstream simplification uses parents '+str(t['upstream_core']['reduced_models']['four_parent_core']['parent_neurons'])+
        f' while clamping all other incoming parents to discovery means. It preserves the AND-like sign pattern in '
        f'{t["upstream_core"]["reduced_models"]["four_parent_core"]["AND_sign_pattern_pct"]:.2f}% of held-out states '
        f'(original neuron: {t["upstream_core"]["original_AND_sign_pattern_pct"]:.2f}%). '
        'The carrier-only and coupled-parent-only reductions do not preserve the same pattern. '
        '[Activation patterns](trace/upstream_core_activation_patterns.pdf) and [complete reduced-core results](trace/upstream_core.json). '
        'This is a reduced path into one neuron, not a replacement for the whole value function.','',
        '[Single-neuron statistics](single_neuron_ablation.csv), [incoming-edge rankings](trace/incoming_edge_rankings.csv), '
        '[trace summary](trace/summary.json), [full summary](summary.json), [protocol](protocol.json).','',report['limitations'],'',
        'Reproduce in jax-v100-cuda12:','', '```bash',
        'XLA_PYTHON_CLIENT_PREALLOCATE=false python concept_mapping/analyze_critic_neuron_transfer_and_trace.py','```',''])
    (out/'README.md').write_text('\n'.join(lines))


if __name__=='__main__':main()

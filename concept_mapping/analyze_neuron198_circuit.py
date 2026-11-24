"""Test fixed incoming circuits to critic neuron 198 using saved activations.

Reuses the original discovery/held-out episode split and cached forward passes.
Only messages on selected L2 -> L3-198 edges are changed. All other final-hidden
neurons, actor outputs, observations and checkpoint parameters remain untouched.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path('concept_mapping/runs/diamond_critic_circuit_10b')
TRACE = ROOT / 'followup_neuron_specificity_and_trace/trace'
Q = np.array([1., -1., 1., -1.])  # A: both; B: diamond; C: neither; D: pickaxe
ORDER = [0, 1, 3, 2]
LABELS = ['Both', 'Diamond only', 'Pickaxe only', 'Neither']
CORE = [159, 256, 334, 461]
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate(h):
    return (h[0] > 0) & (h[1] < 0) & (h[2] < 0) & (h[3] < 0)


def save(fig, out, name):
    for suffix in ('pdf', 'png'):
        fig.savefig(out / f'{name}.{suffix}', dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'neuron198_connection_tests')
    parser.add_argument('--bootstrap-draws', type=int, default=10000)
    parser.add_argument('--random-draws', type=int, default=200)
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--skip-examples', action='store_true')
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    source = TRACE/'diamond_parent_activations.npz'
    with np.load(source) as d:
        parents = d['second_hidden'].astype(np.float64)
        weights = d['incoming_weights'].astype(np.float64)
        bias = float(d['incoming_bias'])
        discovery = d['discovery_indices']
        test = d['heldout_indices']
        original_z = d['z198']
        original_h = d['h198']
    n = len(test)
    assert len(discovery) == 2438 and n == 1045
    assert not set(discovery).intersection(test)
    x = parents[:, test]
    reference = parents[:, discovery].mean(axis=(0, 1))
    z = np.einsum('cni,i->cn', x, weights) + bias
    h = np.tanh(z)
    np.testing.assert_allclose(z, original_z[:, test], atol=1e-10)
    np.testing.assert_allclose(h, original_h[:, test], atol=1e-10)
    rows = read_csv(ROOT/'followup_neuron_specificity_and_trace/per_state/diamond.csv')
    split = read_csv(ROOT/'episode_split.csv')
    split_ids = {int(r['state_index']): (int(r['seed']), int(r['episode_id'])) for r in split}
    assert [(int(r['seed']), int(r['episode_id'])) for r in rows] == [split_ids[int(i)] for i in test]
    assert len(set(split_ids[int(i)] for i in test)) == n
    values = np.array([[float(r['V_'+key]) for r in rows] for key in 'ABCD'])
    for j, key in enumerate('ABCD'):
        np.testing.assert_allclose(h[j], [float(r[f'h198_{key}']) for r in rows], atol=1e-5)
    old_summary = json.loads((TRACE/'summary.json').read_text())
    output_weight = old_summary['output_weight']
    rankings = read_csv(TRACE/'incoming_edge_rankings.csv')
    def ranked(category, count):
        selected = sorted((r for r in rankings if r['category'] == category), key=lambda r: int(r['rank']))
        return [int(r['parent_neuron']) for r in selected[:count]]
    # These selections are fixed by the previous discovery-only analysis.
    assert sorted(set(ranked('upstream_interaction', 2)) | set(ranked('diamond', 2)) | set(ranked('pickaxe', 1))) == CORE
    groups = {
        'diamond_carrier_461': [461],
        'pickaxe_carrier_256': [256],
        'two_carriers': [256, 461],
        'joint_parents': [159, 334],
        'four_parent_core': CORE,
        'diamond_ranked16': ranked('diamond', 16),
        'pickaxe_ranked16': ranked('pickaxe', 16),
    }
    protocol = dict(
        hypothesis='A small incoming circuit combines diamond and pickaxe information to produce an AND-like response in neuron 198 and a positive contribution to critic interaction.',
        source=str(source), source_sha256=sha(source), checkpoint=9999941632,
        discovery_n=len(discovery), heldout_n=n, groups=groups,
        selection='Fixed from previous discovery-only rankings; no fitting or reselection in this follow-up. The held-out set was previously analyzed, so this is mechanistic follow-up, not a fresh untouched replication.',
        reference='Per-parent pooled mean across all four branches of discovery episodes, fixed before held-out evaluation.',
        necessity='Factor-specific donor messages on selected edges only: diamond A->D and B->C receivers (A gets D; B gets C); pickaxe A gets B and D gets C. All other incoming edges and final-hidden neurons remain factual.',
        sufficiency='Only selected incoming edges remain state dependent; all other incoming messages to neuron 198 use fixed discovery means. Other final-hidden neurons remain factual.',
        primary_readouts=['h198 with both present', 'four-condition AND sign pattern', 'h198 interaction', 'value interaction attributable to h198'],
        bootstrap=f'{args.bootstrap_draws:,} paired episode bootstrap draws with the same episode weights across branches and interventions; one state per episode. Conditional on the fixed model and selections.',
        random_control=f'{args.random_draws} uniformly sampled equally sized incoming-edge sets per test; descriptive reference distribution, not an independent statistical population.',
        no_training_or_new_rollouts=True, seed=args.seed,
        scope='L2 messages into one last-hidden neuron and its scalar output edge; upstream parents retain their full original computations. Not a complete input-to-value circuit.'
    )
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print('Loaded cached forward passes and fixed discovery selections.', flush=True)
    intercept = bias + reference @ weights
    def reduced(selected):
        return intercept + np.einsum('cni,i->cn', x[:, :, selected] - reference[selected], weights[selected])
    def patch(selected, factor):
        zz = z.copy()
        pairs = [(0, 3), (1, 2)] if factor == 'diamond' else [(0, 1), (3, 2)]
        for receiver, donor in pairs:
            zz[receiver] += ((x[donor][:, selected] - x[receiver][:, selected]) * weights[selected]).sum(axis=1)
        return zz
    original_I = Q @ values
    original_hI = Q @ h
    candidates = {'original': z, 'mean_clamp_all': reduced([])}
    specs = {'original': ('baseline', '', []), 'mean_clamp_all': ('sufficiency', '', [])}
    for name, selected in groups.items():
        key = f'keep_{name}'
        candidates[key] = reduced(selected)
        specs[key] = ('sufficiency', '', selected)
        for factor in ('diamond', 'pickaxe'):
            key = f'remove_{factor}_via_{name}'
            candidates[key] = patch(selected, factor)
            specs[key] = ('necessity', factor, selected)
    # Endpoint checks ensure edge routing and nonlinear/readout reconstruction.
    np.testing.assert_allclose(reduced(list(range(512))), z, atol=1e-10)
    for factor, donor_pairs in [('diamond', [(0,3),(1,2)]), ('pickaxe', [(0,1),(3,2)])]:
        zz = patch(list(range(512)), factor)
        for receiver, donor in donor_pairs:
            np.testing.assert_allclose(np.tanh(zz[receiver]), h[donor], atol=1e-10)
        np.testing.assert_allclose(Q @ np.tanh(zz), 0, atol=1e-10)
        np.testing.assert_array_equal(patch([], factor), z)
    # Shared bootstrap weights preserve pairing across every compared condition.
    rng = np.random.default_rng(args.seed)
    boot = rng.multinomial(n, np.full(n, 1/n), size=args.bootstrap_draws).astype(np.float64) / n
    boot_hI = boot @ original_hI
    boot_I = boot @ original_I
    summaries = []
    activations = {}; edited_values = {}; metrics = {}
    for name, zz in candidates.items():
        yy = np.tanh(zz)
        vv = values + output_weight * (yy - h)
        hI = Q @ yy
        valueI = Q @ vv
        ownI = output_weight * hI
        # This isolated readout is what the circuit explains, rather than the
        # full value remaining after other 511 last-hidden units are preserved.
        isolated = output_weight * yy
        np.testing.assert_allclose(Q @ isolated, ownI, atol=1e-12)
        np.testing.assert_allclose(valueI, original_I + output_weight*(hI-original_hI), atol=1e-12)
        data = np.column_stack([yy[0], gate(yy), np.abs(yy-h).mean(axis=0), hI,
                                vv[0]-values[0], valueI, ownI, np.abs(hI-original_hI)])
        samples = boot @ data
        ci = np.quantile(samples, [.025, .975], axis=0)
        kind, factor, selected = specs[name]
        row = dict(intervention=name, kind=kind, factor=factor, k=len(selected), n=n,
            hA_mean=float(yy[0].mean()), hA_ci_low=float(ci[0,0]), hA_ci_high=float(ci[1,0]),
            A_positive_pct=float(100*(yy[0]>0).mean()),
            AND_pattern_pct=float(100*gate(yy).mean()), AND_ci_low=float(100*ci[0,1]), AND_ci_high=float(100*ci[1,1]),
            activation_mae=float(np.abs(yy-h).mean()),
            activation_interaction_mean=float(hI.mean()), activation_interaction_ci_low=float(ci[0,3]), activation_interaction_ci_high=float(ci[1,3]),
            activation_interaction_retained_pct=float(100*hI.mean()/original_hI.mean()),
            activation_interaction_retained_ci_low=float(np.quantile(100*samples[:,3]/boot_hI, .025)),
            activation_interaction_retained_ci_high=float(np.quantile(100*samples[:,3]/boot_hI, .975)),
            mean_value_change_A=float((vv[0]-values[0]).mean()), value_change_A_ci_low=float(ci[0,4]), value_change_A_ci_high=float(ci[1,4]),
            full_value_interaction_mean=float(valueI.mean()),
            full_value_interaction_removed_pct=float(100*(original_I-valueI).mean()/original_I.mean()),
            full_value_interaction_removed_ci_low=float(np.quantile(100*(boot_I-samples[:,5])/boot_I,.025)),
            full_value_interaction_removed_ci_high=float(np.quantile(100*(boot_I-samples[:,5])/boot_I,.975)),
            isolated_circuit_value_interaction_mean=float(ownI.mean()),
            isolated_share_of_original_value_interaction_pct=float(100*ownI.mean()/original_I.mean()),
            isolated_share_ci_low=float(np.quantile(100*samples[:,6]/boot_I,.025)),
            isolated_share_ci_high=float(np.quantile(100*samples[:,6]/boot_I,.975)))
        summaries.append(row)
        activations[name] = yy
        edited_values[name] = vv
        metrics[name] = row
    write_csv(out/'intervention_summary.csv', summaries)
    np.savez_compressed(out/'interventions.npz', heldout_indices=test,
        **{f'h__{k}':v for k,v in activations.items()}, **{f'V__{k}':v for k,v in edited_values.items()})
    records = []
    for name in candidates:
        for j, index in enumerate(test):
            r = dict(intervention=name, state_index=int(index), seed=rows[j]['seed'], episode_id=rows[j]['episode_id'])
            for c,key in enumerate('ABCD'):
                r[f'h198_{key}']=float(activations[name][c,j])
                r[f'V_{key}']=float(edited_values[name][c,j])
            r['h198_interaction']=float((Q@activations[name])[j])
            r['value_interaction']=float((Q@edited_values[name])[j])
            records.append(r)
    write_csv(out/'per_state_results.csv', records)
    # Random sets of the same size, shared across factor-removal and retention.
    rng = np.random.default_rng(args.seed+198)
    random_rows = []
    for k in sorted(set(map(len,groups.values()))):
        for draw in range(args.random_draws):
            selected = rng.choice(512,k,replace=False).tolist()
            for mode, factor, zz in [('sufficiency','',reduced(selected))]+[('necessity',f,patch(selected,f)) for f in ('diamond','pickaxe')]:
                yy=np.tanh(zz); hi=Q@yy
                random_rows.append(dict(k=k,draw=draw,kind=mode,factor=factor,parents=';'.join(map(str,selected)),
                    hA_mean=float(yy[0].mean()),A_positive_pct=float(100*(yy[0]>0).mean()),
                    AND_pattern_pct=float(100*gate(yy).mean()),activation_mae=float(np.abs(yy-h).mean()),
                    activation_interaction_retained_pct=float(100*hi.mean()/original_hI.mean()),
                    full_value_interaction_removed_pct=float(100*output_weight*(original_hI-hi).mean()/original_I.mean())))
    write_csv(out/'random_controls.csv', random_rows)
    random_summary=[]
    for k in sorted(set(map(len,groups.values()))):
        for kind,factor in [('sufficiency',''),('necessity','diamond'),('necessity','pickaxe')]:
            rr=[r for r in random_rows if r['k']==k and r['kind']==kind and r['factor']==factor]
            result=dict(k=k,kind=kind,factor=factor,draws=len(rr))
            for metric in ['hA_mean','AND_pattern_pct','activation_mae','activation_interaction_retained_pct','full_value_interaction_removed_pct']:
                a=np.array([r[metric] for r in rr])
                for suffix,val in [('median',np.median(a)),('min',a.min()),('max',a.max()),('q025',np.quantile(a,.025)),('q975',np.quantile(a,.975))]:
                    result[f'{metric}_{suffix}']=float(val)
            random_summary.append(result)
    write_csv(out/'random_control_summary.csv',random_summary)
    # Test sensitivity to the reference used for the other 508 incoming
    # messages. These are fixed discovery-only backgrounds, not fitted biases.
    reference_sensitivity=[]
    for name,ref in [('pooled_discovery',reference),
                     ('neither_discovery',parents[2,discovery].mean(axis=0)),
                     ('both_discovery',parents[0,discovery].mean(axis=0)),
                     ('zero_activation',np.zeros(512))]:
        constant=float(bias+ref@weights-ref[CORE]@weights[CORE])
        zz=constant+np.einsum('cni,i->cn',x[:,:,CORE],weights[CORE])
        yy=np.tanh(zz);hi=Q@yy
        estimates=boot@np.column_stack([gate(yy),hi])
        ci_and=np.quantile(100*estimates[:,0],[.025,.975])
        ci_retained=np.quantile(100*estimates[:,1]/boot_hI,[.025,.975])
        reference_sensitivity.append(dict(reference=name,constant_from_other_edges_and_bias=constant,
            hA_mean=float(yy[0].mean()),AND_pattern_pct=float(100*gate(yy).mean()),
            AND_ci_low=float(ci_and[0]),AND_ci_high=float(ci_and[1]),
            activation_interaction_retained_pct=float(100*hi.mean()/original_hI.mean()),
            retained_ci_low=float(ci_retained[0]),retained_ci_high=float(ci_retained[1]),
            activation_mae=float(np.abs(yy-h).mean())))
    write_csv(out/'reference_sensitivity.csv',reference_sensitivity)
    print('All interventions, paired bootstrap intervals, and random controls complete.',flush=True)
    plt.rcParams.update({'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold',
        'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    plot_reference_sensitivity(out,reference_sensitivity)
    plot_diagram(out, weights, output_weight, intercept, x, h, activations['keep_four_parent_core'])
    plot_results(out, metrics, random_rows, activations, values)
    examples=[]
    if not args.skip_examples:
        examples=render_examples(out,test,activations,edited_values)
    report=dict(protocol=protocol,baseline=metrics['original'],
        primary={key:metrics[key] for key in ['keep_four_parent_core','keep_two_carriers','remove_diamond_via_diamond_carrier_461',
            'remove_pickaxe_via_pickaxe_carrier_256','remove_diamond_via_diamond_ranked16','remove_pickaxe_via_pickaxe_ranked16']},
        mean_original_value_interaction=float(original_I.mean()),output_weight_198=output_weight,
        reference_constant=float(intercept),random_control_summary=random_summary,examples=examples,
        reference_sensitivity=reference_sensitivity,
        validation=dict(cached_forward_max_error=float(np.max(np.abs(z-original_z[:,test]))),
            all_512_edges_reproduce_original=True,all_512_donor_edges_reproduce_donors=True,
            all_512_factor_removals_zero_target_interaction=True,identity_patches_exact=True,
            scalar_readout_effects_verified=True,disjoint_episode_split=True,one_state_per_heldout_episode=True,
            cache_unmodified=sha(source)==protocol['source_sha256']),
        limitations=['One trained checkpoint; candidate choices are exploratory but fixed before this follow-up.',
            'Same previously examined held-out episodes, not a new untouched test population.',
            'Only L2-to-198 incoming messages and the 198-to-output edge are traced; computing the parents still requires the rest of the original network.',
            'The diamond/pickaxe top-16 pathways overlap and contain mixed-context signals.',
            'Four-edge sufficiency depends on the fixed background messages: 99.5% AND-pattern preservation under pooled discovery means falls to 12.7% under neither-condition discovery means.',
            'Mean clamping and donor patches can create internal states outside the natural distribution.',
            'No actor behavior, policy improvement, environmental causal claim, or universal semantic label is tested.'])
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    write_report(out,report)
    print(json.dumps(dict(primary=report['primary'],validation=report['validation']),indent=2),flush=True)


def plot_reference_sensitivity(out,rows):
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),constrained_layout=True)
    labels=['Pooled discovery','Neither discovery','Both discovery','Zero activation']
    for ax,metric,label in zip(axes,['AND_pattern_pct','activation_interaction_retained_pct'],
                              ['AND-like sign pattern (%)','Original h198 interaction retained (%)']):
        ax.bar(range(4),[r[metric] for r in rows],color=['#2563eb','#d97706','#0f766e','#6b7280'])
        ax.set_xticks(range(4),labels,rotation=20,ha='right');ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    axes[0].set_title('Four-edge sufficiency depends on background')
    axes[1].set_title('Same four edges; different fixed references')
    save(fig,out,'reference_sensitivity')


def plot_diagram(out, weights, readout, intercept, x, h, reduced):
    fig,axes=plt.subplots(1,2,figsize=(12,4.9),constrained_layout=True)
    ax=axes[0];ax.set(xlim=(0,10),ylim=(0,7));ax.axis('off')
    ax.set_title('Fixed incoming circuit to neuron 198')
    labels={159:'159: joint context',256:'256: pickaxe signal',334:'334: joint context',461:'461: inverse diamond'}
    for j,i in enumerate(CORE):
        y=6.1-j*1.45
        ax.add_patch(FancyBboxPatch((.1,y-.36),3.5,.72,boxstyle='round,pad=.08',facecolor='#eff6ff',edgecolor='#2563eb'))
        ax.text(1.85,y,labels[i],ha='center',va='center',fontsize=8,fontweight='bold')
        ax.add_patch(FancyArrowPatch((3.7,y),(6.55,3.9),arrowstyle='-|>',mutation_scale=15,
            linewidth=1+abs(weights[i]),color='#2563eb' if weights[i]>0 else '#dc2626',connectionstyle='arc3,rad=0.02'))
        ax.text(4.55,.62*y+1.38,f'{weights[i]:+.3f}',fontsize=9,fontweight='bold',bbox=dict(facecolor='white',edgecolor='none',pad=1))
    ax.add_patch(FancyBboxPatch((6.6,3.2),3.1,1.4,boxstyle='round,pad=.08',facecolor='#fef3c7',edgecolor='#b45309'))
    ax.text(8.15,3.9,'L3 neuron 198\nh = tanh(z)',ha='center',va='center',fontweight='bold')
    ax.text(6.65,2.3,f'Fixed reference input: {intercept:+.3f}\nOther 508 incoming edges clamped',fontsize=9,fontweight='bold')
    ax.text(.15,.08,f'Contribution to value: {readout:+.4f} × h198\nLabels describe responses on this probe bank; upstream computations are retained.',fontsize=8,fontweight='bold')
    matrix=np.array([x[:,:,i].mean(axis=1) for i in CORE]+[h.mean(axis=1),reduced.mean(axis=1)])[:,ORDER]
    aa=axes[1];pic=aa.imshow(matrix,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
    aa.set_xticks(range(4),LABELS,rotation=20,ha='right')
    aa.set_yticks(range(6),[f'L2 {i}' for i in CORE]+['L3 198: original','L3 198: four edges'])
    aa.set_title('Held-out mean activations (1,045 episodes)')
    for i in range(6):
        for j in range(4):aa.text(j,i,f'{matrix[i,j]:+.2f}',ha='center',va='center',fontweight='bold',color='white' if abs(matrix[i,j])>.6 else 'black')
    fig.colorbar(pic,ax=aa,shrink=.8,label='Post-tanh activation')
    save(fig,out,'circuit_diagram')


def plot_results(out,metrics,random_rows,activations,values):
    keys=['original','remove_diamond_via_diamond_carrier_461','remove_pickaxe_via_pickaxe_carrier_256',
        'remove_diamond_via_joint_parents','remove_diamond_via_diamond_ranked16','remove_pickaxe_via_pickaxe_ranked16','keep_four_parent_core']
    labels=['Original','Remove D via 461','Remove P via 256','Remove D via 159/334',
        'Remove D via top 16','Remove P via top 16','Keep four-parent core']
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    m=[metrics[k] for k in keys];means=np.array([r['hA_mean'] for r in m])
    err=np.array([[r['hA_mean']-r['hA_ci_low'] for r in m],[r['hA_ci_high']-r['hA_mean'] for r in m]])
    axes[0].barh(range(len(keys)),means,xerr=err,color=['#6b7280']+['#dc2626']*5+['#2563eb'],capsize=3)
    axes[0].set_yticks(range(len(keys)),labels);axes[0].invert_yaxis();axes[0].axvline(0,color='black',lw=.8)
    axes[0].set(xlim=(-1.2,1.2),xlabel='Mean h198 when both are present (95% CI)',title='Does removing a pathway switch off neuron 198?')
    for j,r in enumerate(m):axes[0].text(.04,j,f'{r["A_positive_pct"]:.1f}% positive',va='center',fontsize=8,fontweight='bold',bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    heat=np.array([activations[k].mean(axis=1)[ORDER] for k in keys])
    pic=axes[1].imshow(heat,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
    axes[1].set_xticks(range(4),LABELS,rotation=20,ha='right');axes[1].set_yticks(range(len(keys)),labels,fontsize=8)
    axes[1].set_title('All four conditions after each intervention')
    for i in range(len(keys)):
        for j in range(4):axes[1].text(j,i,f'{heat[i,j]:+.2f}',ha='center',va='center',fontweight='bold',color='white' if abs(heat[i,j])>.6 else 'black')
    fig.colorbar(pic,ax=axes[1],shrink=.8,label='Mean activation')
    save(fig,out,'necessity_tests')
    suff=['keep_two_carriers','keep_joint_parents','keep_four_parent_core'];sl=['256 + 461','159 + 334','Four-parent core']
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,metric,ylabel in zip(axes,['AND_pattern_pct','activation_interaction_retained_pct','activation_mae'],
        ['AND-like sign pattern (%)','Original h198 interaction retained (%)','Activation MAE']):
        for j,key in enumerate(suff):
            r=metrics[key];rr=[d[metric] for d in random_rows if d['kind']=='sufficiency' and d['k']==r['k']]
            ax.scatter(np.full(len(rr),j)+np.linspace(-.12,.12,len(rr)),rr,s=5,color='#9ca3af',alpha=.35)
            ax.scatter(j,r[metric],s=65,color='#2563eb',zorder=3)
        ax.set_xticks(range(3),sl,rotation=20,ha='right');ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2)
    axes[0].set_title('Four-condition response');axes[1].set_title('Interaction preservation');axes[2].set_title('Numerical fidelity')
    axes[0].text(.02,.97,'Blue: candidate circuit\nGray: 200 random sets per size',transform=axes[0].transAxes,va='top',fontsize=8,fontweight='bold')
    save(fig,out,'sufficiency_and_random_controls')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4),constrained_layout=True)
    keys=['original','keep_four_parent_core','remove_diamond_via_diamond_ranked16','remove_pickaxe_via_pickaxe_ranked16']
    labels=['Original neuron 198','Four-parent circuit','Remove diamond path','Remove pickaxe path']
    means=np.array([metrics[k]['isolated_circuit_value_interaction_mean'] for k in keys])
    axes[0].bar(range(4),means,color=['#6b7280','#2563eb','#dc2626','#d97706'])
    axes[0].set_xticks(range(4),labels,rotation=20,ha='right');axes[0].set_ylabel('Value interaction through neuron 198')
    axes[0].set_title('Contribution of this path alone');axes[0].axhline(0,color='black',lw=.8)
    core=metrics['keep_four_parent_core']['isolated_circuit_value_interaction_mean'];full=(Q@values).mean()
    axes[1].bar([0,1],[full,core],color=['#6b7280','#2563eb'])
    axes[1].set_xticks([0,1],['Entire critic','Four edges → 198 → value']);axes[1].set_ylabel('Mean diamond–pickaxe value interaction')
    axes[1].set_title(f'Four-edge readout accounts for {100*core/full:.1f}%')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    save(fig,out,'effect_on_value')


def render_examples(out,test,activations,values):
    from PIL import Image,ImageDraw,ImageFont,JpegImagePlugin,PdfImagePlugin
    from craftax.craftax_classic.constants import load_all_textures
    from concept_mapping.visualize_symbolic_trajectory import render_local_view
    # Reuse the five earlier random held-out examples; no outcome filtering.
    previous=ROOT/'counterfactual_grids_layer3_top4'
    with np.load(previous/'counterfactual_observations.npz') as d:
        obs=d['observations'];indices=d['state_indices']
    position={int(index):j for j,index in enumerate(test)}
    textures=load_all_textures(7);panel_textures=load_all_textures(28)
    font=ImageFont.truetype(FONT,17);title=ImageFont.truetype(FONT,18)
    destination=out/'examples';destination.mkdir(exist_ok=True)
    sheets=[];records=[]
    for number,index in enumerate(indices,1):
        j=position[int(index)];panels=[]
        for view_idx,c in enumerate(ORDER):
            view=render_local_view(obs[number-1,view_idx],block_size=7,include_inventory=True,display_scale=4,textures=textures,panel_textures=panel_textures)
            panel=Image.new('RGB',(view.width,view.height+115),'white');panel.paste(view,(0,28))
            draw=ImageDraw.Draw(panel);draw.text((view.width/2,2),LABELS[view_idx],anchor='mt',font=title,fill='black')
            lines=[f'value: {values["original"][c,j]:.4f} → {values["keep_four_parent_core"][c,j]:.4f}',
                f'h198: {activations["original"][c,j]:+.3f} → {activations["keep_four_parent_core"][c,j]:+.3f}',
                'original → four edges']
            for k,line in enumerate(lines):draw.text((3,view.height+34+k*25),line,font=font,fill='black' if k!=1 else '#1d4ed8')
            panels.append(panel)
        width,height=panels[0].size
        sheet=Image.new('RGB',(2*width+18,2*height+18),'white')
        for k,panel in enumerate(panels):sheet.paste(panel,((k%2)*(width+18),(k//2)*(height+18)))
        filename=f'example_{number:02d}_sufficiency.png';sheet.save(destination/filename);sheets.append(sheet)
        records.append(dict(state_index=int(index),file=filename,heldout_position=j))
        # Identical input observation in all panels: only internal edge messages change.
        cases=[('Original','original'),('Diamond path removed','remove_diamond_via_diamond_ranked16'),
            ('Pickaxe path removed','remove_pickaxe_via_pickaxe_ranked16'),('Four-edge circuit','keep_four_parent_core')]
        picture=render_local_view(obs[number-1,0],block_size=7,include_inventory=True,display_scale=4,textures=textures,panel_textures=panel_textures)
        width,height=picture.width,picture.height+105
        sheet=Image.new('RGB',(4*width+3*18,height),'white')
        for k,(label,name) in enumerate(cases):
            panel=Image.new('RGB',(width,height),'white');panel.paste(picture,(0,30));draw=ImageDraw.Draw(panel)
            draw.text((width/2,2),label,anchor='mt',font=font,fill='black')
            draw.text((3,picture.height+35),f'value: {values[name][0,j]:.4f}',font=title,fill='black')
            draw.text((3,picture.height+61),f'h198: {activations[name][0,j]:+.4f}',font=title,fill='#1d4ed8')
            sheet.paste(panel,(k*(width+18),0))
        sheet.save(destination/f'example_{number:02d}_necessity.png');sheets.append(sheet)
    sheets[0].save(destination/'all_examples.pdf',save_all=True,append_images=sheets[1:])
    write_csv(destination/'selection.csv',records)
    return records


def write_report(out,r):
    m=r['primary'];baseline=r['baseline'];core=m['keep_four_parent_core']
    lines=['# Connection tests for the diamond–pickaxe response in neuron 198','',
        'Frozen regular feedforward PPO critic, standard rewards, training seed 50, checkpoint 9,999,941,632. '
        'No new training or rollouts. Uses 2,438 discovery episodes for the existing selections/replacement means and '
        'the same previously analyzed 1,045 held-out episodes for evaluation. One four-condition set per episode.','',
        'Hypothesis: a compact set of incoming messages combines diamond availability and iron-pickaxe possession into an '
        'AND-like response in neuron 198, contributing positively to the critic’s conditional valuation.','',
        'A = diamond + iron pickaxe; B = diamond without pickaxe; C = neither; D = pickaxe without diamond. '
        'The activation sign criterion is hA>0 and hB,hC,hD<0. The interaction is A−B+C−D. '
        'Signs describe an operational probe response, not an exact Boolean gate for arbitrary observations.','',
        '## Intervention protocol','',
        'For necessity, replace only selected incoming-edge messages by those from the same state with the relevant factor removed. '
        'Diamond: A receives D messages and B receives C. Pickaxe: A receives B and D receives C. '
        'Other edges into 198 and all other final-hidden neurons stay factual. For sufficiency, retain the selected incoming messages '
        'and fix every other incoming message into 198 to its pooled discovery mean. The parents still use the full upstream network.','',
        'The fixed four-parent circuit is 159, 256, 334, 461 → neuron 198 → scalar value. '
        'The broader factor pathways use the previous discovery-ranked top 16 incoming edges. '
        'These pathways overlap and are not disjoint concept modules. All four core parents are fixed from the earlier discovery analysis.','',
        '## Results','',
        '| Intervention | Mean h198 with both present | Both-present activation positive | Four-condition AND pattern |',
        '| --- | ---: | ---: | ---: |',
        f'| Original | {baseline["hA_mean"]:+.3f} | {baseline["A_positive_pct"]:.2f}% | {baseline["AND_pattern_pct"]:.2f}% |']
    names={'remove_diamond_via_diamond_carrier_461':'Remove diamond message through 461',
        'remove_pickaxe_via_pickaxe_carrier_256':'Remove pickaxe message through 256',
        'remove_diamond_via_diamond_ranked16':'Remove diamond pathway (16 edges)',
        'remove_pickaxe_via_pickaxe_ranked16':'Remove pickaxe pathway (16 edges)',
        'keep_two_carriers':'Keep only 256 and 461','keep_four_parent_core':'Keep four-parent core'}
    for key,label in names.items():
        a=m[key];lines.append(f'| {label} | {a["hA_mean"]:+.3f} | {a["A_positive_pct"]:.2f}% | {a["AND_pattern_pct"]:.2f}% |')
    lines.extend(['',f'The four-parent core retains **{core["activation_interaction_retained_pct"]:.2f}%** '
        f'of neuron 198’s mean activation interaction (95% paired episode-bootstrap CI '
        f'{core["activation_interaction_retained_ci_low"]:.2f}–{core["activation_interaction_retained_ci_high"]:.2f}%). '
        f'Its activation MAE is {core["activation_mae"]:.4f}. '
        'The two proposed detector neurons alone do not recreate the full AND-like response; joint-context parents matter.','',
        '### Sensitivity to the fixed background','',
        'The reported four-edge sufficiency uses the pooled discovery mean for the other 508 incoming messages. '
        'Changing that background is a substantial intervention. Keeping the same four edges but fixing the other messages '
        'to discovery means from the neither condition lowers the AND-pattern rate to 12.7%; using both-present means '
        'gives 99.9%, and zeroing the other messages gives 0%. The compact circuit therefore needs supporting background '
        'input and is not a standalone Boolean gate. Necessity donor patches do not use a mean-clamped background.','',
        '| Background for other incoming messages | AND-like pattern | Target interaction retained |',
        '| --- | ---: | ---: |']+
        [f'| {a["reference"]} | {a["AND_pattern_pct"]:.2f}% | {a["activation_interaction_retained_pct"]:.2f}% |' for a in r['reference_sensitivity']]+['',
        f'The entire critic’s mean value interaction is **{r["mean_original_value_interaction"]:.4f}**. '
        f'Neuron 198’s original isolated readout contributes {baseline["isolated_circuit_value_interaction_mean"]:.4f} '
        f'({baseline["isolated_share_of_original_value_interaction_pct"]:.2f}%). '
        f'The four-parent circuit’s isolated readout contributes **{core["isolated_circuit_value_interaction_mean"]:.4f}**, '
        f'or **{core["isolated_share_of_original_value_interaction_pct"]:.2f}%** of the whole critic interaction '
        f'(95% CI {core["isolated_share_ci_low"]:.2f}–{core["isolated_share_ci_high"]:.2f}%). '
        'Keeping all other final-hidden neurons active is not evidence that these four edges explain their contributions.','',
        '## Interpretation','',
        'Evidence supports a compact contributing incoming circuit for neuron 198 on this probe distribution, conditional on a supporting background. '
        'The pickaxe carrier has a strong causal effect along its connection into 198. '
        'The single diamond carrier has a smaller effect; diamond-dependent joint-context messages are required as well. '
        'Consequently, the computation is not adequately described as one independent diamond detector plus one independent pickaxe detector. '
        'This experiment locates a late-stage computation and its output contribution; it does not yet trace a complete observation-to-value circuit.','',
        '## Controls and artifacts','',
        'All 512-edge retention reconstructs the original target activation. All 512-edge donor patches reconstruct the corresponding donor '
        'target activation and remove the factor interaction. Empty patches are identity operations. '
        'Scalar value changes are checked against the exact linear readout. Source cached activations remain unchanged. '
        'Random controls sample 200 equally sized edge sets for each relevant size. Bootstrap draws preserve pairing across all four conditions.','',
        '- [Circuit diagram](circuit_diagram.pdf)',
        '- [Necessity tests](necessity_tests.pdf)',
        '- [Sufficiency and random controls](sufficiency_and_random_controls.pdf)',
        '- [Contribution to value](effect_on_value.pdf)',
        '- [Background reference sensitivity](reference_sensitivity.pdf)',
        '- [Five example pairs of figures](examples/all_examples.pdf)',
        '- [Intervention summary](intervention_summary.csv), [per-state results](per_state_results.csv)',
        '- [Random controls](random_controls.csv), [validation and full report](summary.json)','',
        '## Limits','']+[f'- {s}' for s in r['limitations']]+['',
        'Reproduce:','', '```bash',
        'OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR=/tmp/mpl-circuit python concept_mapping/analyze_neuron198_circuit.py',
        '```',''])
    (out/'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()

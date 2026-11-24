"""Stimulate L2 neuron 256 while the iron pickaxe remains absent.

Uses the previously sampled 1,000 natural-state pairs. Replace only the
post-tanh activation of neuron 256, then recompute the full last hidden layer
and scalar output. No training, rollouts, or changes to checkpoint weights.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from concept_mapping.plot_collectability_emergence import read_rows,_load_wandb_config,_init_network,_restore_train_state,make_craftax_env_from_name
from concept_mapping.probe_frozen_policy_collectability import write_csv,component_arrays,fingerprint
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


ROOT=Path('concept_mapping/runs/diamond_critic_circuit_10b/random_natural_pickaxe_probe')
RUN='wandb/run-20260630_214658-m0mw4end'
STEP=9999941632


def mean_ci(a,seed,draws=10000):
    rng=np.random.default_rng(seed);means=[]
    for start in range(0,draws,100):
        ix=rng.integers(0,len(a),(min(100,draws-start),len(a)))
        means.extend(a[ix].mean(axis=1).tolist())
    return np.quantile(means,[.025,.975]).tolist()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args();out=args.out or args.source/'neuron256_stimulation'
    out.mkdir(parents=True,exist_ok=True)
    source=read_rows(args.source/'per_state_results.csv')
    meta=json.loads((args.source/'metadata.json').read_text())
    with np.load(args.source/'counterfactual_observations.npz') as saved:obs=saved['observations']
    assert obs.shape==(len(source),2,1345) and np.all(obs[:,1,1331]==0)
    cfg=_load_wandb_config(RUN);env=make_craftax_env_from_name(cfg['ENV_NAME'],auto_reset=False)
    net=_init_network(cfg,env,env.default_params)
    state,_=_restore_train_state(RUN,cfg,net,env,env.default_params,STEP)
    p=state.params['params'];original_hash=fingerprint(component_arrays(state.params,range(8)))
    @jax.jit
    def capture(x):
        h=jnp.tanh(x@p['Dense_4']['kernel']+p['Dense_4']['bias'])
        h2=jnp.tanh(h@p['Dense_5']['kernel']+p['Dense_5']['bias'])
        h3=jnp.tanh(h2@p['Dense_6']['kernel']+p['Dense_6']['bias'])
        v=(h3@p['Dense_7']['kernel']+p['Dense_7']['bias']).squeeze(-1)
        return h2,h3,v
    @jax.jit
    def stimulate(h2,level):
        changed=h2.at[:,256].set(level)
        h3=jnp.tanh(changed@p['Dense_6']['kernel']+p['Dense_6']['bias'])
        v=(h3@p['Dense_7']['kernel']+p['Dense_7']['bias']).squeeze(-1)
        return changed,h3,v
    def batched(fn,*arrays):
        chunks=[fn(*[jnp.asarray(x[start:start+512]) for x in arrays]) for start in range(0,len(arrays[0]),512)]
        return tuple(np.concatenate([np.asarray(c[j]) for c in chunks]) for j in range(3))
    hp,h3p,vp=batched(capture,obs[:,0]);hn,h3n,vn=batched(capture,obs[:,1])
    for values,column in [(vp,'value_with_pickaxe'),(vn,'value_without_pickaxe'),(hp[:,256],'h256_with_pickaxe'),(hn[:,256],'h256_without_pickaxe')]:
        np.testing.assert_allclose(values,[float(r[column]) for r in source],atol=1e-5,rtol=1e-5)
    identity,identity_h3,identity_v=batched(stimulate,hn,hn[:,256])
    np.testing.assert_array_equal(identity,hn)
    np.testing.assert_allclose(identity_v,vn,atol=1e-5,rtol=1e-5)
    matched,matched_h3,matched_v=batched(stimulate,hn,hp[:,256])
    keep=np.arange(512)!=256
    np.testing.assert_array_equal(matched[:,keep],hn[:,keep])
    np.testing.assert_array_equal(matched[:,256],hp[:,256])
    levels=np.array([-1.,-.5,0.,.5,1.],dtype=np.float32)
    dose_h3=[];dose_v=[]
    for level in levels:
        _,h3,v=batched(stimulate,hn,np.full(len(source),level,dtype=np.float32))
        dose_h3.append(h3);dose_v.append(v)
    dose_h3=np.stack(dose_h3);dose_v=np.stack(dose_v)
    rows=[]
    for i,r in enumerate(source):
        row=dict(r)
        row.update(h256_matched_patch=float(matched[i,256]),h198_matched_patch=float(matched_h3[i,198]),
            value_matched_patch=float(matched_v[i]),delta_value_matched_patch=float(matched_v[i]-vn[i]),
            value_clamp_plus1=float(dose_v[-1,i]),delta_value_clamp_plus1=float(dose_v[-1,i]-vn[i]),
            h198_clamp_plus1=float(dose_h3[-1,i,198]))
        for j,level in enumerate(levels):row[f'value_h256_{float(level):g}']=float(dose_v[j,i])
        rows.append(row)
    write_csv(out/'per_state_results.csv',rows)
    summaries=[];groups={}
    for name,mask in [('all',np.ones(len(source),bool)),('no_diamond_visible',np.array([int(r['visible_diamonds'])==0 for r in source])),('diamond_visible',np.array([int(r['visible_diamonds'])>0 for r in source]))]:
        if not mask.any():continue
        groups[name]=mask
        for label,v,h in [('matched_patch',matched_v,matched_h3),('clamp_plus1',dose_v[-1],dose_h3[-1])]:
            delta=v[mask]-vn[mask];true_delta=vp[mask]-vn[mask]
            ci=mean_ci(delta,50)
            summaries.append(dict(group=name,mode=label,n=int(mask.sum()),baseline_no_pickaxe_mean=float(vn[mask].mean()),
                stimulated_mean=float(v[mask].mean()),mean_delta=float(delta.mean()),median_delta=float(np.median(delta)),
                mean_delta_ci_low=float(ci[0]),mean_delta_ci_high=float(ci[1]),
                fraction_value_decreased_pct=float(100*(delta<0).mean()),fraction_value_increased_pct=float(100*(delta>0).mean()),
                mean_actual_inventory_effect=float(true_delta.mean()),
                ratio_of_mean_patch_to_mean_inventory_effect=float(delta.mean()/true_delta.mean()) if abs(true_delta.mean())>1e-8 else None,
                mean_h198_before=float(h3n[mask,198].mean()),mean_h198_after=float(h[mask,198].mean())))
    write_csv(out/'summary.csv',summaries)
    w=np.asarray(p['Dense_7']['kernel']).reshape(-1).astype(np.float64)
    contribution=(matched_h3.astype(np.float64)-h3n.astype(np.float64))*w
    np.testing.assert_allclose(contribution.sum(axis=1),matched_v-vn,atol=1e-5,rtol=1e-5)
    contributions=[]
    for group,mask in groups.items():
        for j in range(512):contributions.append(dict(group=group,neuron=j,mean_value_change=float(contribution[mask,j].mean()),mean_absolute_value_change=float(np.abs(contribution[mask,j]).mean())))
    write_csv(out/'downstream_neuron_contributions.csv',contributions)
    np.savez_compressed(out/'stimulation_results.npz',baseline_values=vn,matched_values=matched_v,
        actual_pickaxe_values=vp,levels=levels,dose_values=dose_v,h256_baseline=hn[:,256],
        h256_matched=hp[:,256],last_hidden_baseline=h3n,last_hidden_matched=matched_h3)
    plt.rcParams.update({'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','font.size':10,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(6.3,4),constrained_layout=True)
    dose_rows=[]
    for group,color in [('no_diamond_visible','#2563eb'),('diamond_visible','#dc2626')]:
        mask=groups[group];delta=dose_v[:,mask]-vn[mask]
        ax.plot(levels,delta.mean(axis=1),marker='o',color=color,label=f'{group.replace("_"," ").capitalize()} (n={int(mask.sum())})')
        for j,level in enumerate(levels):dose_rows.append(dict(group=group,n=int(mask.sum()),activation=float(level),mean_delta_value=float(delta[j].mean()),sd_delta_value=float(delta[j].std())))
    ax.set(xlabel='Imposed L2 neuron 256 activation',ylabel='Mean value change from no-pickaxe state',title='Neuron 256 stimulation; inventory remains empty')
    ax.axhline(0,color='black',lw=.8);ax.grid(alpha=.2);ax.legend(frameon=False,prop={'size':8,'weight':'bold'})
    fig.savefig(out/'activation_response.pdf');fig.savefig(out/'activation_response.png',dpi=250);plt.close(fig)
    write_csv(out/'activation_response.csv',dose_rows)
    selected=meta['example_indices']
    render(out,obs,rows,selected)
    assert fingerprint(component_arrays(state.params,range(8)))==original_hash
    report=dict(source=str(args.source),run=RUN,checkpoint=STEP,n_states=len(source),
        primary='Set only post-tanh L2 neuron 256 in the no-iron-pickaxe branch to its matched factual pickaxe-present activation; recompute all L3 neurons and V.',
        secondary='Clamp the same neuron to +1; also sweep [-1,-0.5,0,0.5,1]. Inventory and all other L2 activations remain unchanged.',
        groups=summaries,example_indices=selected,
        validation=dict(identity_patch_max_error=float(np.max(np.abs(identity_v-vn))),
            only_h256_changed=True,iron_pickaxe_stays_absent=True,parameters_unchanged=True,
            exact_downstream_contribution_error=float(np.max(np.abs(contribution.sum(axis=1)-(matched_v-vn))))),
        limitations='Single model and saved episode distribution. Internal hybrid activations may be out of distribution. Visible-diamond groups are observational; one group has only 21 states. No policy or environment behavior is tested. Matched donor is a small negative activation in one atypical state, so +1 stimulation is reported separately.')
    (out/'metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Stimulating neuron 256 without an iron pickaxe','',report['primary'],'',report['secondary'],'',
        'This experiment reuses the exact 1,000 random natural states and five example indices from the previous inventory-removal study. '
        'No checkpoint, observation, action, or actor update occurs. The only internal intervention is to h2[256].','',
        '| Group | Mode | Mean value change | Value decreases |', '| --- | --- | ---: | ---: |']
    for r in summaries:lines.append(f'| {r["group"]} | {r["mode"]} | {r["mean_delta"]:+.4f} | {r["fraction_value_decreased_pct"]:.1f}% |')
    lines.extend(['','[Five matched three-panel examples](examples/all_examples.pdf): no pickaxe, no pickaxe with neuron 256 patched, actual pickaxe-present state. '
        'The first two panels have identical observations. Only the middle panel has an internal activation intervention.','',
        '[Activation response](activation_response.pdf), [per-state results](per_state_results.csv), [summary](summary.csv), '
        '[downstream contribution decomposition](downstream_neuron_contributions.csv).','', report['limitations'],'',
        'Reproduce in jax-v100-cuda12:','', '```bash',
        'XLA_PYTHON_CLIENT_PREALLOCATE=false python concept_mapping/stimulate_neuron256_pickaxe_probe.py','```',''])
    (out/'README.md').write_text('\n'.join(lines))
    print(json.dumps(report,indent=2),flush=True)


def render(out,obs,rows,selected):
    from PIL import Image,ImageDraw,ImageFont,JpegImagePlugin,PdfImagePlugin
    from craftax.craftax_classic.constants import load_all_textures
    from concept_mapping.visualize_symbolic_trajectory import render_local_view
    destination=out/'examples';destination.mkdir(exist_ok=True)
    textures=load_all_textures(7);panel_textures=load_all_textures(28)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',21)
    title=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',18)
    sheets=[];selection=[]
    for number,i in enumerate(selected,1):
        row=rows[i];panels=[]
        conditions=[('no pickaxe',obs[i,1],row['value_without_pickaxe'],row['h256_without_pickaxe'],row['h198_without_pickaxe']),
            ('no pickaxe\n+ neuron 256 patched',obs[i,1],row['value_matched_patch'],row['h256_matched_patch'],row['h198_matched_patch']),
            ('pickaxe present',obs[i,0],row['value_with_pickaxe'],row['h256_with_pickaxe'],row['h198_with_pickaxe'])]
        for label,observation,value,h256,h198 in conditions:
            value,h256,h198=map(float,(value,h256,h198))
            view=render_local_view(observation,block_size=7,include_inventory=True,display_scale=4,textures=textures,panel_textures=panel_textures)
            panel=Image.new('RGB',(view.width,view.height+145),'white');panel.paste(view,(0,50))
            draw=ImageDraw.Draw(panel)
            for title_line, title_text in enumerate(label.split('\n')):
                draw.text((view.width/2,3+title_line*22),title_text,anchor='mt',font=title,fill='black')
            for line,(label,color) in enumerate([(f'value: {value:.4f}','black'),(f'L2 256: {h256:+.4f}','#9a3412'),(f'L3 198: {h198:+.4f}','#1d4ed8')]):
                draw.text((4,view.height+56+line*27),label,font=font,fill=color)
            panels.append(panel)
        sheet=Image.new('RGB',(3*panels[0].width+36,panels[0].height),'white')
        for j,panel in enumerate(panels):sheet.paste(panel,(j*(panel.width+18),0))
        filename=f'example_{number:02d}.png';sheet.save(destination/filename);sheets.append(sheet)
        selection.append(dict(file=filename,sample_index=i,**row))
    sheets[0].save(destination/'all_examples.pdf',save_all=True,append_images=sheets[1:])
    write_csv(destination/'selection.csv',selection)


if __name__=='__main__':main()

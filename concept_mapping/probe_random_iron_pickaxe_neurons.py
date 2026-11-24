"""Test fixed neurons on random natural iron-pickaxe states and removal edits.

Episode-balanced sampling from saved data, independent of diamond visibility
or collection timing. Original discovery episodes are excluded. No new rollouts.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from concept_mapping.plot_collectability_emergence import (
    read_rows,_load_wandb_config,_init_network,_restore_train_state,make_craftax_env_from_name,
)
from concept_mapping.probe_frozen_policy_collectability import write_csv,component_arrays,fingerprint
import jax
import jax.numpy as jnp
import numpy as np


ROOT=Path('concept_mapping/runs/diamond_critic_circuit_10b')
RUN='wandb/run-20260630_214658-m0mw4end'
CHECKPOINT=9999941632


def sample_states(out,count,seed):
    split=read_rows(ROOT/'episode_split.csv')
    discovery={(int(r['seed']),int(r['episode_id'])) for r in split if r['split']=='discovery'}
    heldout={(int(r['seed']),int(r['episode_id'])) for r in split if r['split']=='heldout'}
    bank_ids=discovery|heldout
    episodes=[]
    for collection_seed,folder in [(50,'master_symbolic_10b_5000episodes'),(51,'master_symbolic_10b_5000episodes_seed51')]:
        directory=Path('concept_mapping/runs')/folder
        info=json.loads((directory/'dataset_metadata.json').read_text())
        assert info['checkpoint']==CHECKPOINT and info['run_path']==RUN
        episodes.extend((collection_seed,int(path.stem.split('_')[-1]),path) for path in sorted((directory/'episodes').glob('episode_*.npz')))
    assert len(episodes)==10000
    other=[e for e in episodes if e[:2] not in bank_ids]
    rng_pool=np.random.default_rng(seed+256)
    extra=set(rng_pool.choice(len(other),int(.3*len(other)),replace=False).tolist())
    pool=[e for e in episodes if e[:2] in heldout]+[e for i,e in enumerate(other) if i in extra]
    assert not discovery & {e[:2] for e in pool}
    rng=np.random.default_rng(seed)
    observations=[];records=[];scanned=0
    for position in rng.permutation(len(pool)):
        collection_seed,episode,path=pool[position]
        scanned+=1
        with np.load(path) as data:
            inventory=data['inventory']
            eligible=np.flatnonzero(inventory[:,8]>0)
            if not len(eligible):continue
            step=int(rng.choice(eligible))
            obs=data['observations'][step].astype(np.float32)
            assert obs[1331]>0
            visible=int(np.sum(obs[:1323].reshape(63,21)[:,10]>.5))
            observations.append(obs)
            records.append(dict(seed=collection_seed,episode_id=episode,timestep=step,
                episode_length=len(inventory),iron_pickaxe_count=int(inventory[step,8]),
                wood_pickaxe_count=int(inventory[step,6]),stone_pickaxe_count=int(inventory[step,7]),
                visible_diamonds=visible,health=int(round(float(obs[1335])*10)),
                source_episode=str(path)))
        if len(records)%100==0:print(f'Sampled {len(records)}/{count} natural states',flush=True)
        if len(records)==count:break
    assert len(records)==count, f'Only {len(records)} eligible episodes in the evaluation pool'
    assert len({(r['seed'],r['episode_id']) for r in records})==count
    np.savez_compressed(out/'sampled_states.npz',observations=np.stack(observations))
    write_csv(out/'sampled_states.csv',records)
    return np.stack(observations),records,dict(total_saved_episodes=len(episodes),original_discovery_episodes_excluded=len(discovery),
        original_heldout_diamond_episodes=len(heldout),additional_heldout_episodes=len(extra),evaluation_pool_size=len(pool),
        scanned_episodes=scanned,sampling='Random episode order; one uniformly random timestep with natural iron pickaxe > 0 per eligible episode. Episode-balanced, not uniform over transitions.',
        pool='Original held-out diamond episodes plus a random 30% of episodes outside the diamond bank. No selection by diamond visibility, collection timing, activation, or value.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=1000)
    parser.add_argument('--examples',type=int,default=5)
    parser.add_argument('--seed',type=int,default=50)
    parser.add_argument('--out',type=Path,default=ROOT/'random_natural_pickaxe_probe')
    args=parser.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    begin=time.monotonic()
    protocol=dict(run=RUN,checkpoint=CHECKPOINT,sampling_seed=args.seed,target_states=args.count,
        neuron_discovery='Fixed L2 neuron 256 and L3 neuron 198 from the previous diamond analysis; no reselection.',
        intervention='Natural iron-pickaxe inventory count retained versus set to zero; all other observation entries unchanged.',
        no_new_rollouts_or_training=True)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    factual,rows,sampling=sample_states(out,args.count,args.seed)
    absent=factual.copy();absent[:,1331]=0
    assert np.all(np.count_nonzero(factual!=absent,axis=1)==1)
    assert np.array_equal(factual[:,:1331],absent[:,:1331]) and np.array_equal(factual[:,1332:],absent[:,1332:])
    cfg=_load_wandb_config(RUN)
    env=make_craftax_env_from_name(cfg['ENV_NAME'],auto_reset=False)
    net=_init_network(cfg,env,env.default_params)
    state,_=_restore_train_state(RUN,cfg,net,env,env.default_params,CHECKPOINT)
    p=state.params['params'];before_hash=fingerprint(component_arrays(state.params,range(8)))
    @jax.jit
    def inspect(obs):
        h=jnp.tanh(obs@p['Dense_4']['kernel']+p['Dense_4']['bias'])
        h2=jnp.tanh(h@p['Dense_5']['kernel']+p['Dense_5']['bias'])
        h3=jnp.tanh(h2@p['Dense_6']['kernel']+p['Dense_6']['bias'])
        v=(h3@p['Dense_7']['kernel']+p['Dense_7']['bias']).squeeze(-1)
        return jnp.stack([h2[:,256],h3[:,198],v],axis=1)
    outputs=[]
    for obs in [factual,absent]:
        outputs.append(np.concatenate([np.asarray(inspect(jnp.asarray(obs[start:start+512]))) for start in range(0,len(obs),512)]).astype(np.float64))
    plus,minus=outputs
    original=np.asarray(net.apply(state.params,jnp.asarray(factual[:64]))[1])
    np.testing.assert_allclose(plus[:64,2],original,atol=1e-5,rtol=1e-5)
    for i,row in enumerate(rows):
        for j,name in enumerate(['h256','h198','value']):
            row[name+'_with_pickaxe']=float(plus[i,j]);row[name+'_without_pickaxe']=float(minus[i,j]);row['delta_'+name]=float(plus[i,j]-minus[i,j])
    write_csv(out/'per_state_results.csv',rows)
    np.savez_compressed(out/'counterfactual_observations.npz',observations=np.stack([factual,absent],axis=1))
    groups=[]
    for name,mask in [('all',np.ones(len(rows),bool)),('diamond_visible',np.array([r['visible_diamonds']>0 for r in rows])),('no_diamond_visible',np.array([r['visible_diamonds']==0 for r in rows]))]:
        if not mask.any():continue
        a,b=plus[mask],minus[mask]
        result=dict(group=name,n=int(mask.sum()),
            mean_h256_with=float(a[:,0].mean()),mean_h256_without=float(b[:,0].mean()),
            mean_delta_h256=float((a[:,0]-b[:,0]).mean()),h256_correct_sign_flip_pct=float(100*((a[:,0]>0)&(b[:,0]<0)).mean()),
            h256_increased_pct=float(100*(a[:,0]>b[:,0]).mean()),
            mean_h198_with=float(a[:,1].mean()),mean_h198_without=float(b[:,1].mean()),
            mean_value_with=float(a[:,2].mean()),mean_value_without=float(b[:,2].mean()),
            mean_delta_value=float((a[:,2]-b[:,2]).mean()),pickaxe_value_positive_pct=float(100*(a[:,2]>b[:,2]).mean()))
        groups.append(result)
    write_csv(out/'summary.csv',groups)
    assert fingerprint(component_arrays(state.params,range(8)))==before_hash
    selected=np.random.default_rng(args.seed+1).choice(len(rows),min(args.examples,len(rows)),replace=False)
    render_pairs(out,factual,absent,rows,selected)
    report=dict(protocol=protocol,sampling=sampling,groups=groups,example_indices=selected.tolist(),
        validation=dict(parameters_unchanged=True,source_parameter_sha256=before_hash,
            pure_inventory_edit=True,manual_value_reference_max_error=float(np.max(np.abs(plus[:64,2]-original)))),
        elapsed_seconds=time.monotonic()-begin,
        limitations='Natural states are sampled conditional on iron pickaxe possession and the saved policy distribution. Episode-balanced sampling is not transition-uniform. Visible-diamond groups are observational. Responses do not establish a globally exclusive semantic label for a neuron.')
    (out/'metadata.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Random natural-state iron-pickaxe removal probe','',
        f'{len(rows):,} natural states, one per randomly ordered eligible episode, sampled at a uniformly random iron-pickaxe-equipped timestep. '
        'No filtering by diamond visibility or first-collection timing. The original neuron-discovery episodes are excluded.', '',
        sampling['pool'],'',sampling['sampling'],'',
        'Only inventory entry 1331 (iron pickaxe) is set to zero. Health, map, other inventory, direction, light and sleeping remain unchanged. '
        'No neuron activation is artificially clamped in this experiment. Same final regular-PPO checkpoint 9,999,941,632.', '',
        '| Group | States | Mean h256 with | Mean h256 without | Correct sign flip |', '| --- | ---: | ---: | ---: | ---: |']
    for r in groups:lines.append(f'| {r["group"]} | {r["n"]} | {r["mean_h256_with"]:+.4f} | {r["mean_h256_without"]:+.4f} | {r["h256_correct_sign_flip_pct"]:.1f}% |')
    lines.extend(['','[Five random visualization pairs](examples/all_examples.pdf). Each pair displays original/removed iron pickaxe, value, L2 neuron 256 and L3 neuron 198. '
        'Examples are selected uniformly from the analyzed sample, independently of the outcome.', '',
        '[Per-state results](per_state_results.csv), [summary](summary.csv), [metadata](metadata.json).', '', report['limitations'], '',
        'Reproduce in jax-v100-cuda12:', '', '```bash',
        'XLA_PYTHON_CLIENT_PREALLOCATE=false python concept_mapping/probe_random_iron_pickaxe_neurons.py', '```',''])
    (out/'README.md').write_text('\n'.join(lines))
    print(json.dumps(report,indent=2),flush=True)


def render_pairs(out,plus,minus,rows,selected):
    from PIL import Image,ImageDraw,ImageFont,JpegImagePlugin,PdfImagePlugin
    from craftax.craftax_classic.constants import load_all_textures
    from concept_mapping.visualize_symbolic_trajectory import render_local_view
    dest=out/'examples';dest.mkdir(exist_ok=True)
    textures=load_all_textures(7);panel_textures=load_all_textures(28)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',22)
    title=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',20)
    sheets=[];selections=[]
    for number,index in enumerate(selected,1):
        row=rows[index];panels=[]
        for label,obs,suffix in [('iron pickaxe',plus[index],'with_pickaxe'),('no iron pickaxe',minus[index],'without_pickaxe')]:
            view=render_local_view(obs,block_size=7,include_inventory=True,display_scale=4,textures=textures,panel_textures=panel_textures)
            panel=Image.new('RGB',(view.width,view.height+125),'white');panel.paste(view,(0,30))
            draw=ImageDraw.Draw(panel);draw.text((view.width/2,3),label,anchor='mt',font=title,fill='black')
            for line,(text,color) in enumerate([(f"value: {row['value_'+suffix]:.4f}",'black'),
                (f"L2 256: {row['h256_'+suffix]:+.4f}",'#9a3412'),(f"L3 198: {row['h198_'+suffix]:+.4f}",'#1d4ed8')]):
                draw.text((4,view.height+36+line*28),text,font=font,fill=color)
            panels.append(panel)
        sheet=Image.new('RGB',(panels[0].width*2+18,panels[0].height),'white')
        sheet.paste(panels[0],(0,0));sheet.paste(panels[1],(panels[0].width+18,0))
        filename=f'example_{number:02d}.png';sheet.save(dest/filename);sheets.append(sheet)
        selections.append(dict(file=filename,sample_index=int(index),**row))
    sheets[0].save(dest/'all_examples.pdf',save_all=True,append_images=sheets[1:])
    write_csv(dest/'selection.csv',selections)


if __name__=='__main__':main()

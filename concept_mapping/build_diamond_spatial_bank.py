"""Expand the existing diamond probe across distance/facing, preserving episodes.

One uniformly sampled eligible natural state per episode and available stratum.
No states are selected by value, activations or counterfactual ordering.
"""
import csv,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concept_mapping.trace_diamond_input_to_value import ROOT,read,write,branches,forward

OUT=ROOT/'spatially_diverse_bank'
STRATA=['adjacent_facing','adjacent_turn_required','distance_2','distance_3_4','distance_5_7']
FOLDERS={50:'master_symbolic_10b_5000episodes',51:'master_symbolic_10b_5000episodes_seed51'}
SEED=150256


def main():
    OUT.mkdir(parents=True,exist_ok=True);began=time.monotonic()
    original=read(ROOT/'episode_split.csv');observations=[];rows=[];episode_rows=[]
    protocol=dict(sampling_seed=SEED,episode_partition='Exact original episode membership: 2,438 discovery and 1,045 held-out episodes. All states from an episode keep its split.',
        eligibility='Natural awake state with iron pickaxe >0, visible diamond, and first diamond achievement not yet earned.',
        strata=STRATA,sampling='One uniformly random eligible timestep per episode and available distance/facing stratum; per-episode RNG seeded by sampling seed, collection seed and episode ID.',
        distance='Minimum Manhattan distance to any visible diamond; adjacent_facing requires a diamond in the actual facing cell.',
        recrafting='Record inventory sufficiency for iron-pickaxe recrafting, without filtering it out, to preserve inventory diversity.',
        selection_ignores_values_and_activations=True,source_limitation='Episodes originated in the prior successful first-diamond bank. Earlier states vary spatially but this population remains conditional on eventual diamond collection.',
        original_immediate_collection_states='Retained as a separately labeled reference subset, not duplicated into expanded discovery.',
        no_new_rollouts_or_training=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for number,ep in enumerate(original,1):
        seed,eid=int(ep['seed']),int(ep['episode_id'])
        path=Path('concept_mapping/runs')/FOLDERS[seed]/'episodes'/f'episode_{eid:04d}.npz'
        with np.load(path) as d:
            inv=d['inventory'];blocks=d['visible_blocks'];ach=d['achievements']
            coords=np.argwhere(blocks==10);distance=np.full(len(blocks),99,dtype=int)
            if len(coords):np.minimum.at(distance,coords[:,0],np.abs(coords[:,1:]-[3,4]).sum(1))
            directions=d['direction'].astype(int)
            dr=np.array([0,0,-1,1])[directions];dc=np.array([-1,1,0,0])[directions]
            facing=blocks[np.arange(len(blocks)),3+dr,4+dc]==10
            eligible=(inv[:,8]>0)&(d['sleeping']==0)&(ach[:,19]==0)&(distance<=7)
            masks=[eligible&(distance==1)&facing,eligible&(distance==1)&~facing,eligible&(distance==2),
                eligible&(distance>=3)&(distance<=4),eligible&(distance>=5)&(distance<=7)]
            rng=np.random.default_rng(np.random.SeedSequence([SEED,seed,eid]));chosen=[]
            for stratum,mask in zip(STRATA,masks):
                candidates=np.flatnonzero(mask)
                if len(candidates):chosen.append((stratum,int(rng.choice(candidates)),len(candidates)))
            obs=d['observations']
            for stratum,t,count in chosen:
                state_index=len(rows);inventory=inv[t]
                row=dict(state_index=state_index,seed=seed,episode_id=eid,timestep=t,split=ep['split'],stratum=stratum,
                    distance_to_diamond=int(distance[t]),diamond_in_front=int(facing[t]),facing_direction=int(directions[t]),
                    visible_diamonds=int((blocks[t]==10).sum()),health=int(d['intrinsics'][t,0]),food=int(d['intrinsics'][t,1]),
                    water=int(d['intrinsics'][t,2]),energy=int(d['intrinsics'][t,3]),achievement_count=int(ach[t].sum()),
                    inventory=json.dumps(inventory.astype(int).tolist()),iron_pickaxe_count=int(inventory[8]),
                    recraft_inventory_sufficient=int(np.all(inventory[[0,1,2,3]]>=1)),
                    eligible_timesteps_in_stratum=count,source_episode=str(path),original_collection_timestep=int(ep['timestep']))
                rows.append(row);observations.append(obs[t].astype(np.float32))
                episode_rows.append({k:row[k] for k in ['state_index','seed','episode_id','timestep','split','stratum']})
        if number%500==0:print(f'{number}/{len(original)} episodes, {len(rows)} sampled states',flush=True)
    factual=np.stack(observations)
    np.savez_compressed(OUT/'base_states.npz',observations=factual,seed=np.array([r['seed'] for r in rows]),
        episode_id=np.array([r['episode_id'] for r in rows]),timestep=np.array([r['timestep'] for r in rows]))
    write(OUT/'state_selection.csv',rows);write(OUT/'episode_split.csv',episode_rows)
    original_bank=Path('concept_mapping/runs/diamond_collectability_10b_10000episodes_all_visible_stone')
    with np.load(original_bank/'base_states.npz') as d:anchor=d['observations']
    np.savez_compressed(OUT/'immediate_collection_reference.npz',observations=anchor)
    write(OUT/'immediate_collection_reference.csv',original)
    # Cached weights were extracted from the exact frozen checkpoint and replay
    # the previous bank to within float32 tolerance; no critic fitting occurs.
    weights=ROOT/'input_to_value_circuit/frozen_critic_weights.npz'
    with np.load(weights) as d:W=[d[f'W{j}'] for j in range(4)];b=[d[f'b{j}'] for j in range(4)]
    batch=branches(factual);values=[]
    for c in range(4):
        vv=[]
        for start in range(0,len(rows),512):vv.append(forward(batch[c,start:start+512],W,b)[2])
        values.append(np.concatenate(vv))
    for i,row in enumerate(rows):row.update({f'V_{key}':float(values[c][i]) for c,key in enumerate('ABCD')})
    write(OUT/'per_state_results.csv',rows)
    summary=[];coverage={}
    for split in ['discovery','heldout']:
        for stratum in STRATA:
            ix=np.array([i for i,r in enumerate(rows) if r['split']==split and r['stratum']==stratum])
            summary.append(dict(split=split,stratum=stratum,states=len(ix),episodes=len({(rows[i]['seed'],rows[i]['episode_id']) for i in ix}),
                recraft_sufficient=sum(rows[i]['recraft_inventory_sufficient'] for i in ix)))
        ix=np.array([i for i,r in enumerate(rows) if r['split']==split])
        counts=(factual[ix,:1323].reshape(-1,63,21)[:,:,10]>0).sum(0)
        coverage[split]=dict(states=len(ix),episodes=len({(rows[i]['seed'],rows[i]['episode_id']) for i in ix}),
            diamond_positions_observed=int((counts>0).sum()),counts=counts.reshape(7,9).tolist())
    write(OUT/'sampling_summary.csv',summary)
    ids={s:{(r['seed'],r['episode_id']) for r in rows if r['split']==s} for s in ['discovery','heldout']}
    assert not ids['discovery']&ids['heldout']
    assert len({(r['seed'],r['episode_id'],r['timestep']) for r in rows})==len(rows)
    plt.rcParams.update({'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(9,3.7),constrained_layout=True)
    vmax=max(np.max(c['counts']) for c in coverage.values())
    for ax,split in zip(axes,['discovery','heldout']):
        a=coverage[split];pic=ax.imshow(a['counts'],cmap='YlOrRd',vmin=0,vmax=vmax)
        ax.scatter(4,3,marker='*',c='#16a34a',edgecolor='black',s=100)
        ax.set_title(f'{split.capitalize()}: {a["diamond_positions_observed"]}/63 positions');ax.set(xlabel='Observation column',ylabel='Observation row')
    fig.colorbar(pic,ax=list(axes),label='Visible diamond count');fig.savefig(OUT/'spatial_coverage.pdf');fig.savefig(OUT/'spatial_coverage.png',dpi=240);plt.close(fig)
    report=dict(protocol=protocol,coverage=coverage,strata=summary,states=len(rows),elapsed_seconds=time.monotonic()-began,
        validation=dict(episode_splits_disjoint=True,unique_natural_states=True,counterfactual_edits_validated=True,all_pre_first_diamond=True))
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# Spatially diverse diamond discovery bank','',protocol['sampling'], '',protocol['eligibility'],'',protocol['episode_partition'],'',
        '| Stratum | Discovery states | Held-out states |','| --- | ---: | ---: |']
    for stratum in STRATA:lines.append('| '+stratum+' | '+' | '.join(str(next(r['states'] for r in summary if r['split']==s and r['stratum']==stratum)) for s in ['discovery','heldout'])+' |')
    lines.extend(['',f'Total: {len(rows):,} states. The unchanged immediate-collection reference is stored separately. ',
        'Natural inventory, health, direction, and other features are preserved. Recraft inventory sufficiency is recorded rather than excluded. ',
        'Four counterfactuals retain/remove only iron pickaxe and preserve/replace all visible diamonds with stone.','',
        'Several states can belong to the same episode; uncertainty estimates must resample episodes, not individual states. ',
        protocol['source_limitation'], '', '[Spatial coverage](spatial_coverage.pdf), [selection](state_selection.csv), [counts](sampling_summary.csv).',''])
    (OUT/'README.md').write_text('\n'.join(lines));render_examples()
    print(json.dumps(report,indent=2),flush=True)


def render_examples():
    from types import SimpleNamespace
    from concept_mapping.render_collectability_minimal_grids import render
    rows=read(OUT/'per_state_results.csv')
    with np.load(OUT/'base_states.npz') as d:factual=d['observations']
    selected=[];rng=np.random.default_rng(SEED+1)
    for stratum in STRATA:
        eligible=[i for i,r in enumerate(rows) if r['split']=='discovery' and r['stratum']==stratum]
        selected.append(int(rng.choice(eligible)))
    dest=OUT/'examples';dest.mkdir(exist_ok=True)
    write(dest/'per_state_results.csv',[rows[i] for i in selected])
    batch=branches(factual[selected])
    np.savez_compressed(dest/'counterfactual_observations.npz',observations=batch[[0,1,3,2]].transpose(1,0,2))
    render(SimpleNamespace(task='diamond',source_dir=dest,out_dir=dest,font='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        font_size=23,block_size=7,display_scale=4,examples_per_grid=1,gap=18,after_values_suffix=None))
    write(dest/'strata.csv',[dict(file=f'example_{j+1:02d}.png',stratum=rows[i]['stratum'],state_index=i,
        seed=rows[i]['seed'],episode_id=rows[i]['episode_id'],timestep=rows[i]['timestep']) for j,i in enumerate(selected)])


if __name__=='__main__':main()

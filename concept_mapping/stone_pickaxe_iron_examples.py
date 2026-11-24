"""Seeded 15-state pilot: stone pickaxe retained/removed x iron/path."""
import csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from concept_mapping.analyze_wood_sword_combat_followup import (
    save_csv,evaluate_values,_load_wandb_config,_init_network,_restore_train_state,make_craftax_env_from_name)
from concept_mapping.visualize_symbolic_trajectory import render_local_view
from concept_mapping.render_water_success_examples import _font
from craftax.craftax_classic.constants import load_all_textures
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent/'runs'
OUT=ROOT/'stone_pickaxe_iron_10b_15examples'


def main():
    OUT.mkdir(exist_ok=True);bank=[];metadata=[];missing=0
    for seed,folder in [(50,'master_symbolic_10b_5000episodes'),(51,'master_symbolic_10b_5000episodes_seed51')]:
        for path in sorted((ROOT/folder/'episodes').glob('episode_*.npz')):
            with np.load(path) as d:
                ach=np.r_[d['achievements'][:,18],d['terminal_achievements'][18]].astype(int)
                hits=np.flatnonzero(np.diff(ach)>0)
                if not len(hits):missing+=1;continue
                t=int(hits[0]);o=d['observations'][t].copy();direction=int(d['direction'][t])
                rr=3+[0,0,-1,1][direction];cc=4+[-1,1,0,0][direction]
                assert d['actions'][t]==5 and not d['sleeping'][t]
                assert d['visible_blocks'][t,rr,cc]==9 and d['inventory'][t,7]>0
                assert d['inventory'][t,3]==0
                # Verify factual post-transition iron inventory increments, including terminal transitions.
                after=d['observations'][t+1] if t+1<len(d['actions']) else d['terminal_observation']
                assert round(float(after[1326])*10)==1
                row=dict(seed=seed,episode_id=int(d['episode_id']),timestep=t,target_row=rr,target_col=cc)
                for k,v in zip(['wood','stone','coal','iron','diamond','sapling','wood_pickaxe','stone_pickaxe','iron_pickaxe','wood_sword','stone_sword','iron_sword'],d['inventory'][t]):row['inventory_'+k]=int(v)
                for k,v in zip(['health','food','water','energy'],d['intrinsics'][t]):row[k]=int(v)
                row['achievement_flags']=json.dumps(d['achievements'][t].astype(int).tolist())
                metadata.append(row);bank.append(o)
    print('Eligible first-iron states:',len(bank),flush=True)
    chosen=np.random.default_rng(50).choice(len(bank),15,replace=False)
    factual=np.stack([bank[i] for i in chosen]);rows=[metadata[i] for i in chosen]
    target=np.array([(r['target_row']*9+r['target_col'])*21 for r in rows])
    np.savez_compressed(OUT/'selected_base_states.npz',observations=factual,target_offset=target)
    run='wandb/run-20260630_214658-m0mw4end';config=_load_wandb_config(run)
    env=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False)
    net=_init_network(config,env,env.default_params)
    state,checkpoint=_restore_train_state(run,config,net,env,env.default_params,9999941632)
    branches={};values={}
    for pickaxe in ['removed','retained']:
        for resource in ['iron','path']:
            obs=factual.copy()
            if pickaxe=='removed':obs[:,1330]=0
            if resource=='path':
                obs[np.arange(15),target+9]=0;obs[np.arange(15),target+7]=1
            for i in range(15):
                keep=np.ones(1345,bool);keep[1330]=False;keep[target[i]+np.array([7,9])]=False
                assert np.array_equal(obs[i,keep],factual[i,keep])
            key=pickaxe+'_'+resource;branches[key]=obs
            values[key]=evaluate_values(state.params,net,obs,512).astype(float)
    fv=evaluate_values(state.params,net,factual,1)
    assert np.allclose(fv,values['retained_iron'],atol=1e-5,rtol=1e-5)
    dp=values['retained_iron']-values['removed_iron'];da=values['retained_path']-values['removed_path'];interaction=dp-da
    textures=load_all_textures(7);panel=load_all_textures(28);sheets=[]
    for i,r in enumerate(rows):
        r.update({'V_'+k:float(v[i]) for k,v in values.items()})
        r.update(pickaxe_effect_iron=float(dp[i]),pickaxe_effect_path=float(da[i]),interaction=float(interaction[i]))
        frames=[]
        for key in ['removed_iron','retained_iron','removed_path','retained_path']:
            view=render_local_view(branches[key][i],block_size=7,include_inventory=True,display_scale=4,textures=textures,panel_textures=panel)
            header=Image.new('RGB',(view.width,94),'white');draw=ImageDraw.Draw(header)
            pick,res=key.split('_')
            draw.text((6,4),'Stone pickaxe' if pick=='retained' else 'No stone pickaxe',fill='black',font=_font(20))
            draw.text((6,33),'Iron present' if res=='iron' else 'Iron replaced by path',fill='black',font=_font(18,False))
            draw.text((6,61),f'V = {values[key][i]:.4f}',fill='black',font=_font(21))
            frame=Image.new('RGB',(view.width,view.height+94),'white');frame.paste(header,(0,0));frame.paste(view,(0,94));frames.append(frame)
        w,h=frames[0].size;sheet=Image.new('RGB',(2*w+20,2*h+190),'white');draw=ImageDraw.Draw(sheet)
        draw.text((8,5),f"Seed {r['seed']} | episode {r['episode_id']} | step {r['timestep']}",fill='black',font=_font(17))
        draw.text((8,31),'Pre-first-iron collection; original health',fill='black',font=_font(17,False))
        for j,f in enumerate(frames):sheet.paste(f,((j%2)*(w+20),60+(j//2)*(h+20)))
        y=80+2*h
        for dy,label,v in [(0,'Pickaxe effect, iron',dp[i]),(27,'Pickaxe effect, path',da[i]),(54,'Interaction',interaction[i])]:draw.text((8,y+dy),f'{label}: {v:+.4f}',fill='black',font=_font(18))
        name=f'example_{i+1:02d}_seed{r["seed"]}_episode{r["episode_id"]}.png';sheet.save(OUT/name);r['image']=name;sheets.append(sheet)
    sheets[0].save(OUT/'fifteen_stone_pickaxe_iron_examples.pdf',save_all=True,append_images=sheets[1:])
    save_csv(OUT/'per_state_results.csv',rows)
    summary=dict(eligible_episodes=len(bank),excluded_no_first_iron=missing,sample_size=15,seed=50,checkpoint=checkpoint,
        mean_pickaxe_effect_iron=float(dp.mean()),mean_pickaxe_effect_path=float(da.mean()),mean_interaction=float(interaction.mean()),median_interaction=float(np.median(interaction)),positive_count=int((interaction>0).sum()),
        design='One pre-first-iron-collection state per eligible episode; unfiltered random 15; stone pickaxe factual count or zero, front iron tile or PATH. All other observation coordinates unchanged; no actions, rewards or achievement changes.',
        limitation='Exploratory 15-state pilot, not full-bank inference. Iron-to-path also changes walkability; tests tool-resource valuation, not MAKE_STONE_PICKAXE action value.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

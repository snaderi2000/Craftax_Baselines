"""Five random examples per first-event collectability test, frozen critic."""
import argparse,csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw
from concept_mapping.analyze_wood_sword_combat_followup import (
    save_csv,evaluate_values,_load_wandb_config,_init_network,_restore_train_state,make_craftax_env_from_name)
from concept_mapping.visualize_symbolic_trajectory import render_local_view
from concept_mapping.render_water_success_examples import _font
from craftax.craftax_classic.constants import load_all_textures

ROOT=Path(__file__).resolve().parent/'runs'
OUT=ROOT/'collectability_three_tasks_10b_pilot'
SPECS={'diamond':(19,10,8,4),'iron':(18,9,7,4),'plant':(11,16,None,15)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--iron-count',type=int,default=5)
    args=parser.parse_args()
    if args.iron_count<5:raise ValueError('Keep at least the original five examples')
    OUT.mkdir(exist_ok=True)
    previous={}
    for task in SPECS:
        path=OUT/task/'per_state_results.csv'
        if path.exists():
            with path.open() as f:
                previous[task]=[(int(r['seed']),int(r['episode_id']),int(r['timestep'])) for r in csv.DictReader(f)]
    candidates={k:[] for k in SPECS};counts={k:dict(no_event=0,can_recraft=0,invalid=0,eligible=0) for k in SPECS}
    for seed,folder in [(50,'master_symbolic_10b_5000episodes'),(51,'master_symbolic_10b_5000episodes_seed51')]:
        for p in sorted((ROOT/folder/'episodes').glob('episode_*.npz')):
            with np.load(p) as d:
                achievements=d['achievements'];terminal=d['terminal_achievements'];inv=d['inventory'];actions=d['actions']
                for task,(achievement,block,tool,neutral) in SPECS.items():
                    ts=np.flatnonzero(np.diff(np.r_[achievements[:,achievement],terminal[achievement]].astype(int))>0)
                    if not len(ts):counts[task]['no_event']+=1;continue
                    t=int(ts[0]);inventory=inv[t]
                    recraft=(inventory[0]>=1 and inventory[1]>=1 and (task=='iron' or (inventory[3]>=1 and inventory[2]>=1)))
                    if tool is not None and recraft:counts[task]['can_recraft']+=1;continue
                    dr=int(d['direction'][t]);rr=3+[0,0,-1,1][dr];cc=4+[-1,1,0,0][dr]
                    if actions[t]!=5 or d['sleeping'][t] or d['visible_blocks'][t,rr,cc]!=block or (tool is not None and inventory[tool]==0):
                        counts[task]['invalid']+=1;continue
                    candidates[task].append((seed,p,t,rr,cc))
                    counts[task]['eligible']+=1
    print(json.dumps(counts),flush=True)
    run='wandb/run-20260630_214658-m0mw4end';cfg=_load_wandb_config(run)
    env=make_craftax_env_from_name('Craftax-Classic-Symbolic-v1',auto_reset=False)
    net=_init_network(cfg,env,env.default_params)
    state,ckpt=_restore_train_state(run,cfg,net,env,env.default_params,9999941632)
    textures=load_all_textures(7);panel=load_all_textures(28)
    all_sheets=[];summary={}
    for task,(achievement,block,tool,neutral) in SPECS.items():
        out=OUT/task;out.mkdir(exist_ok=True);rows=[];sheets=[];saved=[]
        pool=candidates[task];choice=np.random.default_rng(50).choice(len(pool),min(5,len(pool)),replace=False)
        if task=='iron' and args.iron_count>5:
            remaining=np.setdiff1d(np.arange(len(pool)),choice)
            extra=np.random.default_rng(51).choice(remaining,args.iron_count-5,replace=False)
            choice=np.concatenate([choice,extra])
        for rank,index in enumerate(choice,1):
            seed,p,t,rr,cc=pool[index]
            with np.load(p) as d:
                factual=d['observations'][t].copy()
                row=dict(seed=seed,episode_id=int(d['episode_id']),timestep=t,target_row=rr,target_col=cc,
                    inventory=json.dumps(d['inventory'][t].astype(int).tolist()),
                    achievement_flags=json.dumps(d['achievements'][t].astype(int).tolist()),
                    health=int(d['intrinsics'][t,0]),food=int(d['intrinsics'][t,1]),water=int(d['intrinsics'][t,2]),energy=int(d['intrinsics'][t,3]))
            offset=(rr*9+cc)*21
            if task in previous and rank<=len(previous[task]):
                assert (seed,row['episode_id'],t)==previous[task][rank-1], 'Selected example changed'
            ore_offsets=np.flatnonzero(factual[:1323].reshape(63,21)[:,:17].argmax(1)==block)*21 if tool is not None else np.array([offset])
            if tool is not None:row['visible_target_ore_tiles']=len(ore_offsets)
            conditions=[('A',True,True),('B',False,True),('D',True,False),('C',False,False)] if tool is not None else [('ripe',True,True),('unripe',True,False)]
            observations=[];labels=[]
            for key,retained,present in conditions:
                obs=factual.copy()
                if tool is not None and not retained:obs[1323+tool]=0
                if not present:
                    obs[ore_offsets+block]=0;obs[ore_offsets+neutral]=1
                keep=np.ones(1345,bool)
                keep[ore_offsets+block]=False;keep[ore_offsets+neutral]=False
                if tool is not None:keep[1323+tool]=False
                assert np.array_equal(obs[keep],factual[keep])
                assert np.argmax(obs[offset:offset+17])==(block if present else neutral)
                if tool is not None and not present:
                    assert not np.any(obs[:1323].reshape(63,21)[:,:17].argmax(1)==block)
                    assert np.all(obs[ore_offsets+neutral]==1)
                    assert np.count_nonzero(obs[:1323]!=factual[:1323])==2*len(ore_offsets)
                if tool is not None:assert obs[1323+tool]==(factual[1323+tool] if retained else 0)
                observations.append(obs)
                labels.append(('Pickaxe retained' if retained else 'Pickaxe removed',task.title()+' present' if present else f'All {task} to stone') if tool is not None else ('Ripe plant' if present else 'Unripe plant','No inventory edits'))
            batch=np.stack(observations);values=evaluate_values(state.params,net,batch,32)
            ref=evaluate_values(state.params,net,factual[None],1)[0]
            assert np.isclose(ref,values[0],atol=1e-5,rtol=1e-5)
            row['V_factual']=float(ref)
            row.update({'V_'+c[0]:float(v) for c,v in zip(conditions,values)})
            if tool is not None:
                a,b,c,dv=[row['V_'+k] for k in 'ABCD']
                row.update(interaction=(a-b)-(dv-c),success=bool(a>max(b,c) and min(b,c)>dv))
                note=f"Interaction: {row['interaction']:+.4f} | ordering: {row['success']}"
            else:
                row.update(ripe_effect=float(values[0]-values[1]),success=bool(values[0]>values[1]));note=f"Ripe - unripe: {row['ripe_effect']:+.4f}"
            frames=[]
            for obs,value,label in zip(observations,values,labels):
                view=render_local_view(obs,block_size=7,include_inventory=True,display_scale=4,textures=textures,panel_textures=panel)
                header=Image.new('RGB',(view.width,94),'white');draw=ImageDraw.Draw(header)
                for y,text,size in [(4,label[0],20),(33,label[1],18),(61,f'V = {value:.4f}',21)]:draw.text((6,y),text,fill='black',font=_font(size))
                frame=Image.new('RGB',(view.width,view.height+94),'white');frame.paste(header,(0,0));frame.paste(view,(0,94));frames.append(frame)
            w,h=frames[0].size;nr=2 if tool is not None else 1
            sheet=Image.new('RGB',(2*w+20,nr*h+(nr-1)*20+110),'white');draw=ImageDraw.Draw(sheet)
            draw.text((8,5),f'{task.title()} | seed {seed} ep {row["episode_id"]} step {t}',fill='black',font=_font(17))
            sampling_seed=51 if task=='iron' and rank>5 else 50
            row['sampling_seed']=sampling_seed
            draw.text((8,31),f'Random seed {sampling_seed}; before first successful event',fill='black',font=_font(16,False))
            for j,f in enumerate(frames):sheet.paste(f,((j%2)*(w+20),60+(j//2)*(h+20)))
            draw.text((8,60+nr*h+(nr-1)*20+8),note,fill='black',font=_font(17))
            file=f'example_{rank:02d}.png';sheet.save(out/file);row['image']=file
            rows.append(row);sheets.append(sheet);all_sheets.append(sheet);saved.append(batch)
        if rows:
            save_csv(out/'per_state_results.csv',rows);np.savez_compressed(out/'counterfactual_observations.npz',observations=np.stack(saved))
            sheets[0].save(out/'five_examples.pdf',save_all=True,append_images=sheets[1:5])
            if len(sheets)>5:
                sheets[0].save(out/f'{len(sheets)}_examples.pdf',save_all=True,append_images=sheets[1:])
                sheets[5].save(out/'ten_additional_examples.pdf',save_all=True,append_images=sheets[6:])
        summary[task]=dict(**counts[task],sampled=len(rows),successes=sum(r['success'] for r in rows))
    if all_sheets:all_sheets[0].save(OUT/'all_pilot_examples.pdf',save_all=True,append_images=all_sheets[1:])
    (OUT/'summary.json').write_text(json.dumps(dict(tasks=summary,checkpoint=ckpt,seed=50,
        recraft_exclusion='Inventory sufficiency irrespective of station access: stone pickaxe needs wood+stone; iron pickaxe needs wood+stone+iron+coal. Conservative exclusion.',
        checks='Pre-first-achievement DO states, awake, facing matching tile, tool present. Non-edited observation coordinates identical; factual batch/single inference agrees within tolerance.',
        additional_iron_sampling_seed=51,
        interpretation='Original five selected states preserved and verified by seed/episode/timestep. Extra iron examples sampled without replacement from remaining eligible episodes with seed 51. All visible target-type ore replaced by STONE, preserving other terrain and creatures. Stone is itself mineable; unripe plants retain future food opportunity. Pilot only, no rollouts/training/time/reward awards.'),indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

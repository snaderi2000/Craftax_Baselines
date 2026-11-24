"""Five seeded random factual-health iron-retention grids, using water layout."""
import csv
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw
from craftax.craftax_classic.constants import load_all_textures
from concept_mapping.render_water_success_examples import _font
from concept_mapping.visualize_symbolic_trajectory import render_local_view

BASE=Path(__file__).resolve().parent/'runs/combat_sword_interaction_10b_10000episodes'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wood',action='store_true')
    parser.add_argument('--health',type=int,choices=[2,9])
    args=parser.parse_args()
    run=BASE/('wood_vs_none_followup' if args.wood else 'iron_retention_health_followup');out=run/'example_grids';out.mkdir(exist_ok=True)
    if args.health is not None:
        assert args.wood, '--health requires --wood'
        run=BASE/'wood_vs_none_health2_vs9'
        out=run/f'example_grids_health{args.health}';out.mkdir(exist_ok=True)
    with (run/'per_state_results.csv').open() as f:rows=list(csv.DictReader(f))
    chosen=np.random.default_rng(50).choice(len(rows),10 if args.wood else 5,replace=False)
    textures=load_all_textures(7);panel=load_all_textures(28)
    selection=[];sheets=[]
    with np.load(BASE/'state_bank.npz') as bank:
        for rank,index in enumerate(chosen,1):
            r=rows[index]
            if args.wood:
                r=dict(r)
                if args.health is not None:
                    for sword in ['none','wood']:
                        for condition in ['present','absent']:
                            r[f'V_{sword}_{condition}']=r[f'V_h{args.health}_{sword}_{condition}']
                    for condition in ['present','absent']:r['delta_'+condition]=r[f'h{args.health}_{condition}']
                    r['interaction']=r[f'h{args.health}_interaction']
                for a,b in [('removed','none'),('retained','wood')]:
                    for c in ['present','absent']:r[f'V_regular_{a}_{c}']=r[f'V_{b}_{c}']
                for c in ['present','absent']:r['regular_'+c]=r['delta_'+c]
                r['regular_interaction']=r['interaction']
            ix=np.flatnonzero((bank['seed']==int(r['seed']))&(bank['episode_id']==int(r['episode_id']))&(bank['timestep']==int(r['timestep'])))
            assert len(ix)==1
            factual=bank['observations'][ix[0]].copy();target=int(bank['target_index'][ix[0]])
            frames=[]
            for iron,condition in [('removed','present'),('retained','present'),('removed','absent'),('retained','absent')]:
                obs=factual.copy()
                if args.health is not None:obs[1335]=args.health/10
                if iron=='removed':obs[1334]=0
                if args.wood:
                    obs[1332:1335]=0
                    obs[1332]=.1 if iron=='retained' else 0
                if condition=='absent':obs[target]=0
                keep=np.ones(len(obs),bool);keep[[1334,target]]=False
                if args.wood:keep[1332:1335]=False
                if args.health is not None:keep[1335]=False
                assert np.array_equal(obs[keep],factual[keep])
                view=render_local_view(obs,block_size=7,include_inventory=True,display_scale=4,textures=textures,panel_textures=panel)
                header=Image.new('RGB',(view.width,94),'white');draw=ImageDraw.Draw(header)
                title=('Wood sword' if iron=='retained' else 'No sword') if args.wood else 'Iron '+iron
                draw.text((8,4),title,fill='black',font=_font(21))
                draw.text((8,33),'Zombie '+condition,fill='black',font=_font(18,False))
                draw.text((8,61),f"V = {float(r[f'V_regular_{iron}_{condition}']):.4f}",fill='black',font=_font(21))
                frame=Image.new('RGB',(view.width,view.height+94),'white');frame.paste(header,(0,0));frame.paste(view,(0,94));frames.append(frame)
            w,h=frames[0].size;gap=20;sheet=Image.new('RGB',(2*w+gap,2*h+190),'white');draw=ImageDraw.Draw(sheet)
            draw.text((8,5),f"Seed {r['seed']} | episode {r['episode_id']} | step {r['timestep']}",fill='black',font=_font(17))
            subtitle='other swords zero' if args.wood else 'wood/stone preserved'
            health_label=f"Factual health {r['health']}" if args.health is None else f'Forced health {args.health}'
            draw.text((8,31),f"{health_label} | {subtitle}",fill='black',font=_font(16,False))
            for j,f in enumerate(frames):sheet.paste(f,((j%2)*(w+gap),60+(j//2)*(h+gap)))
            y=60+2*h+gap
            for offset,label,key in [(0,'Iron effect, zombie present','regular_present'),(27,'Iron effect, zombie absent','regular_absent'),(54,'Interaction','regular_interaction')]:
                if args.wood:label=label.replace('Iron','Wood')
                draw.text((8,y+offset),f'{label}: {float(r[key]):+.4f}',fill='black',font=_font(18))
            name=f'example_{rank:02d}_seed{r["seed"]}_episode{r["episode_id"]}'
            sheet.save(out/(name+'.png'));sheet.save(out/(name+'.pdf'));sheets.append(sheet)
            selection.append(dict(file=name+'.png',sampling_seed=50,selection='unfiltered random',**r))
    sheets[0].save(out/('ten_wood_sword_examples.pdf' if args.wood else 'five_iron_retention_examples.pdf'),save_all=True,append_images=sheets[1:])
    with (out/'selection.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(selection[0]));writer.writeheader();writer.writerows(selection)
    for r in selection:print(r['file'],r['regular_interaction'])


if __name__=='__main__':main()

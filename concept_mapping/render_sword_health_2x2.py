"""Adapt the water-example PIL 2x2 layout for paired sword/target edits."""
import csv
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw
from craftax.craftax_classic.constants import load_all_textures
from concept_mapping.render_water_success_examples import _font
from concept_mapping.visualize_symbolic_trajectory import render_local_view

BASE=Path(__file__).resolve().parent/'runs/combat_sword_interaction_10b_10000episodes'


def main():
    run=BASE/'wood_vs_none_health3_vs9';out=run/'example_grids';out.mkdir(exist_ok=True)
    rows=list(csv.DictReader((run/'per_state_results.csv').open()))
    # An unfiltered, reproducible random example; same factual base for both health settings.
    index=int(np.random.default_rng(50).integers(len(rows)));r=rows[index]
    with np.load(BASE/'state_bank.npz') as bank:
        ix=np.flatnonzero((bank['seed']==int(r['seed']))&(bank['episode_id']==int(r['episode_id']))&(bank['timestep']==int(r['timestep'])))
        assert len(ix)==1
        factual=bank['observations'][ix[0]].copy();target=int(bank['target_index'][ix[0]])
    textures=load_all_textures(7);panel=load_all_textures(28)
    for health in [3,9]:
        frames=[]
        for sword,condition in [('none','present'),('wood','present'),('none','absent'),('wood','absent')]:
            obs=factual.copy();obs[1332:1335]=0;obs[1335]=health/10
            obs[1332]=.1 if sword=='wood' else 0
            if condition=='absent':obs[target]=0
            view=render_local_view(obs,block_size=7,include_inventory=True,display_scale=4,textures=textures,panel_textures=panel)
            header=Image.new('RGB',(view.width,94),'white');draw=ImageDraw.Draw(header)
            draw.text((8,4),'Wood sword' if sword=='wood' else 'No sword',fill='black',font=_font(21))
            draw.text((8,33),'Zombie '+condition,fill='black',font=_font(18,False))
            draw.text((8,61),f"V = {float(r[f'V_h{health}_{sword}_{condition}']):.4f}",fill='black',font=_font(21))
            frame=Image.new('RGB',(view.width,view.height+94),'white');frame.paste(header,(0,0));frame.paste(view,(0,94));frames.append(frame)
        w,h=frames[0].size;gap=20;sheet=Image.new('RGB',(2*w+gap,2*h+180),'white');draw=ImageDraw.Draw(sheet)
        draw.text((8,5),f'Health {health} | seed {r["seed"]}, episode {r["episode_id"]}, step {r["timestep"]}',fill='black',font=_font(17))
        draw.text((8,31),'Random example (seed 50); identical base state',fill='black',font=_font(16,False))
        for j,f in enumerate(frames):sheet.paste(f,((j%2)*(w+gap),60+(j//2)*(h+gap)))
        y=60+2*h+gap
        draw.text((8,y),f"Wood - none, present: {float(r[f'h{health}_present']):+.4f}",fill='black',font=_font(18))
        draw.text((8,y+27),f"Wood - none, absent:  {float(r[f'h{health}_absent']):+.4f}",fill='black',font=_font(18))
        draw.text((8,y+54),f"Interaction: {float(r[f'h{health}_interaction']):+.4f}",fill='black',font=_font(20))
        sheet.save(out/f'example_health{health}_2x2.png');sheet.save(out/f'example_health{health}_2x2.pdf')
    with (out/'selection.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(r));w.writeheader();w.writerow(r)
    print(out)


if __name__=='__main__':main()

"""Render 15 unfiltered random paired inventory swaps using existing textures."""
import csv,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw
from craftax.craftax_classic.constants import load_all_textures
from concept_mapping.render_water_success_examples import _font
from concept_mapping.visualize_symbolic_trajectory import render_local_view

ROOT=Path(__file__).resolve().parent/'runs/early_late_inventory_swap_10b_10000episodes'


def main():
    out=ROOT/'example_grids';out.mkdir(exist_ok=True)
    with (ROOT/'per_episode_results.csv').open() as f:rows=list(csv.DictReader(f))
    chosen=np.random.default_rng(50).choice(len(rows),15,replace=False)
    textures=load_all_textures(7);panel=load_all_textures(28);sheets=[];selection=[]
    with np.load(ROOT/'paired_states.npz') as bank:
        for rank,i in enumerate(chosen,1):
            e=bank['early'][i];l=bank['late'][i];r=rows[i];frames=[]
            for key,base,inventory in [('EE',e,e),('EL',e,l),('LE',l,e),('LL',l,l)]:
                o=base.copy();o[1323:1335]=inventory[1323:1335]
                view=render_local_view(o,block_size=7,include_inventory=True,display_scale=4,textures=textures,panel_textures=panel)
                header=Image.new('RGB',(view.width,94),'white');draw=ImageDraw.Draw(header)
                draw.text((6,4),('Early' if key[0]=='E' else 'Late')+' observation',fill='black',font=_font(20))
                draw.text((6,33),('Early' if key[1]=='E' else 'Late')+' inventory',fill='black',font=_font(18,False))
                draw.text((6,61),f"V = {float(r['V_'+key]):.4f}",fill='black',font=_font(21))
                f=Image.new('RGB',(view.width,view.height+94),'white');f.paste(header,(0,0));f.paste(view,(0,94));frames.append(f)
            w,h=frames[0].size;sheet=Image.new('RGB',(2*w+20,2*h+170),'white');draw=ImageDraw.Draw(sheet)
            draw.text((8,5),f"Seed {r['seed']} | episode {r['episode_id']}",fill='black',font=_font(18))
            draw.text((8,31),f"Early step {r['early_step']} | late step {r['late_step']}",fill='black',font=_font(17,False))
            for j,f in enumerate(frames):sheet.paste(f,((j%2)*(w+20),60+(j//2)*(h+20)))
            y=2*h+80
            draw.text((8,y),f"Late inv effect, early: {float(r['early_context_effect']):+.4f}",fill='black',font=_font(18))
            draw.text((8,y+28),f"Late inv effect, late: {float(r['late_context_effect']):+.4f}",fill='black',font=_font(18))
            filename=f'example_{rank:02d}.png';sheet.save(out/filename);sheets.append(sheet)
            selection.append(dict(file=filename,seed=r['seed'],episode_id=r['episode_id'],sampling_seed=50))
    sheets[0].save(out/'fifteen_inventory_swap_examples.pdf',save_all=True,append_images=sheets[1:])
    with (out/'selection.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(selection[0]));w.writeheader();w.writerows(selection)
    print(out)


if __name__=='__main__':main()

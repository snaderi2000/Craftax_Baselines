"""Create the printable field atlas from the same measured values as the website."""
import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
DIST = HERE / 'gru_atlas/dist'
PAPER, INK, SECOND, LINE = '#fcfcf8', '#102e2b', '#476360', '#bdcbc4'
TEAL, ORANGE = '#08694f', '#c95422'
PALETTE = LinearSegmentedColormap.from_list('occupancy', ['#f0f4e8', '#c3ded1', '#75b6a1', '#328471', '#0b4b42'])


def plot(data, target):
    mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'pdf.fonttype': 42, 'axes.facecolor': PAPER,
                         'text.color': INK, 'axes.labelcolor': SECOND,
                         'xtick.color': SECOND, 'ytick.color': SECOND})
    fig = plt.figure(figsize=(14, 8.2), facecolor=PAPER)
    s = data['summary']

    def text(x,y,txt,size=11,**kw):
        return fig.text(x,y,txt,fontsize=size,va='baseline',**kw)

    def rule(x0,x1,y,color=LINE,lw=.7):
        fig.add_artist(Line2D([x0,x1],[y,y],transform=fig.transFigure,color=color,linewidth=lw))

    text(.05,.947,'CRAFTAX CLASSIC     /     POLICY EVALUATION',10,weight='bold',color=TEAL)
    text(.95,.947,'1,000 PROCEDURAL WORLDS',10,ha='right',color=SECOND)
    text(.05,.873,'Where the GRU plays, and what it misses',31,weight='bold')
    text(.05,.827,'A 10-billion-step recurrent policy receives a 7 × 9 local view of each 64 × 64 world. All episodes start at row 32, column 32.',12)
    rule(.05,.95,.8,TEAL,1.2)

    spatial = [('occupancy','Time spent',f'{s["transitions"]:,} pre-action positions'),
               ('diamonds','Diamond distribution',f'{s["initialDiamondTiles"]:,} diamond tiles at reset')]
    for (key, title, sub), lx, axleft in zip(spatial, [.05,.332], [.089,.371]):
        text(lx,.748,title,16,weight='bold')
        text(lx,.713,sub,10,color=SECOND)
        grid=np.asarray(data['maps'][key]).reshape(8,8,8,8).sum(axis=(1,3))
        share = 100 * grid / grid.sum()
        ax=fig.add_axes([axleft,.322,.214,.365])
        ax.pcolormesh(np.arange(0,65,8),np.arange(0,65,8),share,
                      cmap=PALETTE,norm=Normalize(0,3),edgecolors='#f4f7ed',linewidth=.6)
        ax.set(ylabel='World row',xlabel='World column',xticks=[0,16,32,48,63],yticks=[0,16,32,48,63],
               xlim=(0,64),ylim=(64,0),aspect='equal')
        ax.tick_params(length=0,labelsize=9,pad=4)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel('World column', fontsize=9, labelpad=4)
        ax.set_ylabel('World row', fontsize=9, labelpad=2)
        ax.plot(32.5,32.5,marker='+',mew=3.5,ms=12,color='white')
        ax.plot(32.5,32.5,marker='+',mew=1.8,ms=12,color=ORANGE)

    cbax=fig.add_axes([.075,.211,.287,.014])
    cb=fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0,3),cmap=PALETTE),cax=cbax,
                    ticks=[0,1,2,3],orientation='horizontal',format='%.0f%%')
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2,labelsize=9)
    text(.075,.243,'Share of each panel’s total per 8 × 8 region',10,weight='bold')
    text(.397,.212,'+',19,weight='bold',color=ORANGE)
    text(.417,.214,'Starting tile',10)
    text(.05,.122,'Each square pools the same coordinates across 1,000 independently generated layouts.',10,color=SECOND)

    right=.639
    text(right,.748,'Achievement completion and survival',15,weight='bold')
    text(right,.711,'Episodes in each outcome, out of 1,000',10,color=SECOND)
    xs=[right,.833,.894,.949]
    for xx, t in zip(xs,['Achievements','At limit','Died','Total']):
        text(xx,.654,t,10,weight='bold',ha='left' if xx==right else 'right')
    rule(right,.95,.636)
    for y, lab, count, total in [(.59,'All 22',s['successTimeout'],s['all22']),
                                  (.536,'Fewer than 22',s['timeout']-s['successTimeout'],s['totalFailure'])]:
        text(xs[0],y,lab,11)
        for xx,num in zip(xs[1:],[count,total-count,total]):
            text(xx,y,f'{num:,}',12,ha='right',weight='bold' if xx==xs[-1] else 'normal')
    rule(right,.95,.51)
    text(right,.465,'The 154 diamond misses',15,weight='bold')
    rows=[('Never in local view',s['missNeverSeen'],'#116d58'),
          ('Seen only without an iron pickaxe',s['missSeen']-s['missSeenWithPickaxe'],'#7da99b'),
          ('Seen while holding an iron pickaxe',s['missSeenWithPickaxe'],'#b94c24')]
    for i,(lab,n,col) in enumerate(rows):
        y=.417-i*.068
        text(right,y,lab,10)
        text(.949,y,f'{n}   ({100*n/s["misses"]:.1f}%)',10,ha='right',weight='bold')
        fig.add_artist(Rectangle((right,y-.023),.31,.009,transform=fig.transFigure,facecolor='#e6ede6',edgecolor='none'))
        fig.add_artist(Rectangle((right,y-.023),.31*n/s['misses'],.009,transform=fig.transFigure,facecolor=col,edgecolor='none'))
    text(right,.205,'Across those eight equipped runs, diamond never reached',10,color=SECOND)
    text(right,.181,'a cardinally adjacent tile while the pickaxe was held.',10,color=SECOND)
    rule(.05,.95,.091)
    text(.05,.049,f'Checkpoint {data["checkpoint"]:,}   /   categorical sampling T = {data["temperature"]:.1f}   /   evaluation seed {data["seed"]:,}   /   10,000-step limit',9,color=SECOND)
    text(.95,.049,'CRAFTAX  •  LOCAL GRU',9,ha='right',color=TEAL,weight='bold')
    fig.savefig(target, format='pdf',metadata={'Title':'Where the GRU plays, and what it misses',
                    'Subject':'Local-GRU spatial visitation, initial diamond distribution and outcomes across 1,000 Craftax episodes'})
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DIST/'data.js')
    parser.add_argument('--out',type=Path,default=DIST/'field-atlas.pdf')
    args=parser.parse_args()
    data=json.loads(args.data.read_text().removeprefix('window.ATLAS=').strip().removesuffix(';'))
    plot(data,args.out)

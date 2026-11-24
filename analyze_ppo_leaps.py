"""Analyze the completed screen without pooling selected and held-out results."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parent/'concept_mapping/runs/ppo_sample_efficiency_leaps_1m'
    progress=json.loads((root/'progress.json').read_text())
    assert progress['state']=='complete'
    result=dict(training_steps=progress['actual_training_steps'],
        evaluation_executed_transitions=progress['evaluation_executed_transitions'],phases={})
    for phase,seeds in [('screen',(50,51)),('confirmation',(52,53))]:
        phase_result={}
        for condition in (('baseline','spatial','self_imitation','combined') if phase=='screen' else ('baseline','spatial')):
            summaries=[];stone=0;wood=0;n=0
            for seed in seeds:
                out=root/phase/condition/f'seed{seed}'/'evaluation'
                summaries.append(json.loads((out/'summary.json').read_text()))
                with (out/'episodes.csv').open() as f:episodes=list(csv.DictReader(f))
                n+=len(episodes);stone+=sum(int(r['collect_stone']) for r in episodes)
                wood+=sum(int(r['make_wood_pickaxe']) for r in episodes)
            phase_result[condition]=dict(seeds=list(seeds),evaluation_episodes=n,stone_episodes=stone,
                wood_pickaxe_episodes=wood,mean_return=float(np.mean([d['mean_return'] for d in summaries])),
                mean_achievements=float(np.mean([d['mean_achievements'] for d in summaries])),
                crafter_score=float(np.mean([d['crafter_score'] for d in summaries])))
        b=phase_result['baseline'];s=phase_result['spatial']
        result['phases'][phase]=dict(conditions=phase_result,
            spatial_return_gain=s['mean_return']-b['mean_return'],
            spatial_return_gain_pct=100*(s['mean_return']/b['mean_return']-1),
            stone_success_ratio=s['stone_episodes']/b['stone_episodes'])
    (root/'final_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    plt.rcParams.update({'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    for condition,label,color in [('baseline','PPO','#2563eb'),('spatial','Spatial PPO','#dc2626')]:
        xs=np.array([262144,524288,1048576])/1e6
        for ax,metric,ylabel in [(axes[0],'mean_return','Mean episode return'),
                               (axes[1],'collect_stone','Stone collection (%)')]:
            values=[]
            for seed in (52,53):
                row=[]
                for step in (262144,524288,1048576):
                    suffix='evaluation' if step==1048576 else f'evaluation_{step}'
                    d=json.loads((root/'confirmation'/condition/f'seed{seed}'/suffix/'summary.json').read_text())
                    row.append(d[metric] if metric=='mean_return' else d['achievement_success_pct'][metric])
                values.append(row)
            values=np.array(values)
            ax.plot(xs,values.mean(0),'-o',color=color,label=label,lw=2)
            ax.fill_between(xs,values.min(0),values.max(0),color=color,alpha=.15)
            ax.set(xlabel='Training transitions (millions)',ylabel=ylabel)
            ax.grid(alpha=.2);ax.legend(prop={'weight':'bold'})
    axes[0].set_title('Held-out policy return');axes[1].set_title('Held-out stone collection')
    axes[1].set_ylim(bottom=0)
    fig.supxlabel('Two fresh training seeds; shaded bands show the seed range. No search at evaluation.',fontsize=8,fontweight='bold')
    for ext in ('pdf','png','jpeg'):fig.savefig(root/f'learning_curves.{ext}',dpi=250)
    plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

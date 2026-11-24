"""Aggregate seed-level frozen evaluations as the memory pilot completes."""
import csv,fcntl,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parent/'concept_mapping/runs/ppo_memory_comparison_10m'
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.summary.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data={};rows=[]
        for architecture in ('mlp','gru','transformer'):
            for seed in (50,51,52):
                for budget in (1,5,10):
                    p=root/f'{architecture}_seed{seed}/evaluation_{budget}m/summary.json'
                    if not p.exists():continue
                    d=json.loads(p.read_text());data[(architecture,seed,budget)]=d
                    rows.append(dict(architecture=architecture,seed=seed,budget_millions=budget,
                        checkpoint=d['checkpoint'],mean_return=d['mean_return'],mean_length=d['mean_length'],
                        mean_achievements=d['mean_achievements'],**d['achievement_success_pct']))
        if not rows:return
        with (root/'evaluation_results.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
        for architecture,color in (('mlp','#2563eb'),('gru','#dc2626'),('transformer','#159447')):
            for ax,metric in zip(axes,('mean_return','mean_achievements')):
                x=[];y=[];sd=[]
                for budget in (1,5,10):
                    ds=[data[(architecture,s,budget)] for s in (50,51,52) if (architecture,s,budget) in data]
                    if not ds:continue
                    x.append(np.mean([d['checkpoint'] for d in ds]));v=[d[metric] for d in ds]
                    y.append(np.mean(v));sd.append(np.std(v,ddof=1) if len(v)>1 else 0.)
                if x:
                    ax.plot(x,y,'-o',color=color,label=architecture.upper())
                    ax.fill_between(x,np.array(y)-sd,np.array(y)+sd,color=color,alpha=.15)
        for ax,title in zip(axes,('Frozen-policy return','Frozen-policy achievements')):
            ax.set(title=title,xlabel='Training environment steps');ax.grid(alpha=.2);ax.legend()
        axes[0].set_ylabel('Mean undiscounted return');axes[1].set_ylabel('Achievements per episode')
        for ext in ('pdf','png'):fig.savefig(root/f'learning_curves.{ext}',dpi=200)
        plt.close(fig)
        lines=['# PPO memory comparison: frozen standard-environment evaluation','',
            '128 episodes per policy/checkpoint; evaluation seed 15050. Shading is the sample standard deviation across available training seeds, not a confidence interval. Incomplete groups are provisional.','',
            '| Architecture | Finished final evaluations | Mean return | Mean achievements | Diamond (%) |',
            '|---|---:|---:|---:|---:|']
        for a in ('mlp','gru','transformer'):
            ds=[data[(a,s,10)] for s in (50,51,52) if (a,s,10) in data]
            if ds:lines.append(f"| {a.upper()} | {len(ds)}/3 | {np.mean([d['mean_return'] for d in ds]):.3f} | {np.mean([d['mean_achievements'] for d in ds]):.3f} | {np.mean([d['achievement_success_pct']['collect_diamond'] for d in ds]):.2f} |")
        lines+=['','All 22 achievement rates are in `evaluation_results.csv`. Architecture and parameter-count differences prevent attributing a result solely to memory.']
        (root/'results.md').write_text('\n'.join(lines)+'\n')
        print(f'Updated comparison from {len(rows)} frozen evaluations')


if __name__=='__main__':main()

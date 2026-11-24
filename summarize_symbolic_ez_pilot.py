"""Summarize symbolic learned-model search against the matched PPO controls."""
import argparse,csv,fcntl,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    base=Path(__file__).resolve().parent/'concept_mapping/runs'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=base/'symbolic_ez_10m')
    out=parser.parse_args().study.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'.summary.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        rows=[]
        for architecture in ('ppo','ez_search','ez_policy'):
            for seed in (50,51,52):
                for budget in (1,5,10):
                    if architecture=='ppo':path=base/f'ppo_memory_comparison_10m/mlp_seed{seed}/evaluation_{budget}m/summary.json'
                    else:
                        mode='search' if architecture=='ez_search' else 'policy'
                        path=out/f'seed{seed}'/f'evaluation_{budget}m_{mode}'/'summary.json'
                    if not path.exists():continue
                    d=json.loads(path.read_text())
                    rows.append(dict(architecture=architecture,seed=seed,budget_millions=budget,
                        mean_return=d['mean_return'],mean_length=d['mean_length'],mean_achievements=d['mean_achievements'],
                        **d['achievement_success_pct']))
        if not rows:return
        with (out/'evaluation_results.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
        for name,color in [('ppo','#2563eb'),('ez_search','#dc2626'),('ez_policy','#159447')]:
            for ax,metric in zip(axes,('mean_return','mean_achievements')):
                xs=[];ys=[];ss=[]
                for budget in (1,5,10):
                    paired={r['seed'] for r in rows if r['architecture']=='ez_search' and r['budget_millions']==budget}
                    group=[r[metric] for r in rows if r['architecture']==name and r['budget_millions']==budget and (name!='ppo' or not paired or r['seed'] in paired)]
                    if not group:continue
                    xs.append(budget);ys.append(np.mean(group));ss.append(np.std(group,ddof=1) if len(group)>1 else 0.)
                if xs:
                    ax.plot(xs,ys,'-o',color=color,label=name)
                    ax.fill_between(xs,np.array(ys)-ss,np.array(ys)+ss,color=color,alpha=.15)
        for ax,title in zip(axes,('Frozen-policy return','Frozen-policy achievements')):
            ax.set(title=title,xlabel='Training interactions (millions)');ax.grid(alpha=.2);ax.legend()
        for ext in ('pdf','png'):fig.savefig(out/f'learning_curves.{ext}',dpi=200)
        plt.close(fig)
        lines=['# Symbolic EZ-style pilot: provisional results','',
            'Frozen standard-reset evaluation;128 matched-seed episodes per policy/checkpoint. Means across available training seeds; plot shading is seed standard deviation, not a confidence interval.','',
            '| Budget (M) | Agent | Seeds evaluated | Return | Achievements | Diamond (%) |',
            '|---:|---|---:|---:|---:|---:|']
        for budget in (1,5,10):
            evaluated={r['seed'] for r in rows if r['architecture']=='ez_search' and r['budget_millions']==budget}
            if not evaluated:continue
            for name in ('ppo','ez_search','ez_policy'):
                group=[r for r in rows if r['architecture']==name and r['budget_millions']==budget and r['seed'] in evaluated]
                if group:lines.append(f"| {budget} | {name} | {len(group)} | {np.mean([r['mean_return'] for r in group]):.3f} | {np.mean([r['mean_achievements'] for r in group]):.3f} | {np.mean([r['collect_diamond'] for r in group]):.2f} |")
        lines+=['','Table comparisons restrict PPO to the same training seeds with completed EZ search evaluations at each budget. The CSV retains all available controls.']
        lines+=['','All achievement rates are in `evaluation_results.csv`. Training time and model diagnostics are in each seed directory.']
        (out/'results.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

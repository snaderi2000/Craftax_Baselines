"""Compare completed pilot evaluations and export W&B training curves."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='concept_mapping/runs/cppo_100m_pilot')
    args=p.parse_args();out=Path(args.out)
    conditions=['baseline','frozen_critic','bc_ppo','cppo','shuffled_cppo']
    rows=[]
    for condition in conditions:
        path=out/condition/'evaluation.json'
        if not path.exists():
            raise RuntimeError(f'Awaiting evaluation: {condition}')
        rows.append(dict(condition=condition,**json.loads(path.read_text())))
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    keys=[k for k in rows[0] if k not in ['condition','episodes','checkpoint','evaluation_seed']]
    show=['returned_episode_returns']+[k for k in keys if any(x in k.lower() for x in ['collect_wood','collect_stone','collect_iron','collect_diamond'])]
    fig,axes=plt.subplots(1,len(show),figsize=(4*len(show),4),squeeze=False)
    for ax,k in zip(axes[0],show):
        y=np.array([r[k] for r in rows])
        ax.bar(conditions,y);ax.tick_params(axis='x',rotation=35);ax.set_title(k.replace('Achievements/',''))
    fig.tight_layout();fig.savefig(out/'evaluation_comparison.pdf');fig.savefig(out/'evaluation_comparison.png',dpi=160);plt.close(fig)
    import wandb
    api=wandb.Api(timeout=30)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for c in conditions:
        manifest=json.loads((out/c/'run_manifest.json').read_text())
        run=api.run(manifest['wandb_path'])
        history=list(run.scan_history(keys=['env_step','episode_return','score']))
        if not history:
            history=list(run.scan_history())
        with (out/c/'training_history.json').open('w') as f:json.dump(history,f)
        for ax,k in zip(axes,['episode_return','score']):
            points=[(r.get('env_step',r.get('timestep',r.get('_step',0)*8192)),r[k]) for r in history if k in r and r[k] is not None]
            if points:
                x,y=np.array(points).T
                smoothed=[];v=y[0]
                for value in y:v=.95*v+.05*value;smoothed.append(v)
                ax.plot(x,smoothed,label=c)
                ax.set_xlabel('New environment interactions');ax.set_ylabel(k)
    axes[0].legend();fig.tight_layout();fig.savefig(out/'learning_curves.pdf');fig.savefig(out/'learning_curves.png',dpi=160)
    baseline=rows[0]['returned_episode_returns']
    report=['# C-PPO 100M pilot results','', 'Held-out evaluation: 128 episodes per condition, evaluation seed 1050.','',
            '| Condition | Mean return | Difference from baseline |','|---|---:|---:|']
    report += [f"| {r['condition']} | {r['returned_episode_returns']:.4f} | {r['returned_episode_returns']-baseline:+.4f} |" for r in rows]
    report += ['', 'This is a single-training-seed expert-transfer pilot. The inherited experience and BC updates are additional resources. It does not yet test cumulative improvement over fresh generations.']
    (out/'results.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report))


if __name__=='__main__':main()

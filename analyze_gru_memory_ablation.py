"""Collect the matched GRU/no-history frozen evaluations without GPU work."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np


def collect(study):
    rows=[];summaries={};signatures=[]
    for arm in ('normal','reset_every_step'):
        for path in (study/arm/'gru').glob('evaluation_*/summary.json'):
            try:data=json.loads(path.read_text())
            except json.JSONDecodeError:continue
            label=path.parent.name.removeprefix('evaluation_')
            summaries[arm,label]=(data,path.parent);signatures.append((str(path),path.stat().st_mtime_ns))
            rows.append(dict(condition=arm,budget=label,steps=data['training_steps'],episodes=data['episodes'],
                mean_achievements=data['mean_achievements'],original_return=data['mean_original_return'],
                shaped_return=data['mean_shaped_return'],mean_length=data['mean_length'],
                all22_pct=data['all_22_success_pct'],diamond_pct=data['achievement_success_pct']['collect_diamond'],
                plant_pct=data['achievement_success_pct']['eat_plant'],lava_pct=data['lava_pct']))
    rows.sort(key=lambda r:(r['steps'],r['condition']))
    if not rows:return [],summaries,signatures
    with (study/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# GRU history ablation: frozen evaluations','',
        'One paired training seed. Evaluation uses the same seeded original-generator worlds, normal action sampling and a 10,000-step episode cap. These are frozen evaluations, not training metrics.','',
        '| Steps | Condition | Mean achievements | All 22 (%) | Diamond (%) | Plant (%) | Original return | Mean length |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:lines.append(f"| {r['steps']:,} | {r['condition']} | {r['mean_achievements']:.3f} | {r['all22_pct']:.2f} | {r['diamond_pct']:.2f} | {r['plant_pct']:.2f} | {r['original_return']:.3f} | {r['mean_length']:.1f} |")
    achievements=[]
    for (arm,label),(data,_) in summaries.items():
        for name,rate in data['achievement_success_pct'].items():
            achievements.append(dict(condition=arm,budget=label,steps=data['training_steps'],achievement=name,success_pct=rate))
    with (study/'achievement_rates.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(achievements[0]));writer.writeheader();writer.writerows(achievements)
    if all((arm,'wallclock') in summaries for arm in ('normal','reset_every_step')):
        trials=[]
        for arm in ('normal','reset_every_step'):
            data,folder=summaries[arm,'wallclock']
            with (folder/'episodes.csv').open() as f:trials.append(sorted(csv.DictReader(f),key=lambda r:int(r['episode'])))
        assert [r['episode'] for r in trials[0]]==[r['episode'] for r in trials[1]]
        rng=np.random.default_rng(28051);indices=rng.integers(len(trials[0]),size=(10000,len(trials[0])))
        differences={}
        for key in ('all_22','original_return','length'):
            scale=100 if key=='all_22' else 1
            def value(row):return float(row[key]=='True') if key=='all_22' else float(row[key])
            delta=(np.array([value(r) for r in trials[0]])-np.array([value(r) for r in trials[1]]))*scale
            differences[key]=dict(normal_minus_reset=float(delta.mean()),ci95=np.quantile(delta[indices].mean(1),[.025,.975]).tolist())
        (study/'paired_final_differences.json').write_text(json.dumps(differences,indent=2)+'\n')
        lines+=['','Final paired differences (normal GRU minus reset control):']
        for key,d in differences.items():lines.append(f"- {key}: {d['normal_minus_reset']:.3f}, 95% episode-bootstrap CI [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}].")
        lines+=['','Intervals describe these evaluation worlds conditional on one trained policy pair; they do not measure training-seed variability.']
    (study/'comparison.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold'})
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,(key,title) in zip(axes.flat,[('mean_achievements','Mean achievements'),('original_return','Original return'),('all22_pct','All 22 (%)'),('mean_length','Mean episode length')]):
        for arm,color,label in [('normal','#2462ef','GRU: history enabled'),('reset_every_step','#df2828','GRU: reset every step')]:
            selected=[r for r in rows if r['condition']==arm]
            ax.plot([r['steps']/1e6 for r in selected],[r[key] for r in selected],'-o',color=color,label=label)
        ax.set(title=title,xlabel='Training interactions (millions)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=9);fig.tight_layout()
    for extension in ('pdf','png'):fig.savefig(study/f'learning_curves.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    return rows,summaries,signatures


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--watch',action='store_true');args=p.parse_args();end=time.time()+10*3600
    previous=None
    while True:
        signatures=sorted((str(p),p.stat().st_mtime_ns) for arm in ('normal','reset_every_step')
            for p in (args.study/arm/'gru').glob('evaluation_*/summary.json'))
        if signatures!=previous:
            _,summaries,_=collect(args.study);previous=signatures
        else:summaries={}
        states={}
        for arm in ('normal','reset_every_step'):
            path=args.study/arm/'gru/progress.json'
            states[arm]=json.loads(path.read_text()).get('state') if path.exists() else 'pending'
        done=all(s in ('complete','complete_with_failures','wall_clock_complete','failed') for s in states.values())
        (args.study/'comparison_progress.json').write_text(json.dumps(dict(time=time.time(),state='complete' if done else 'waiting',training_states=states),indent=2)+'\n')
        if not args.watch or done or time.time()>end:break
        time.sleep(30)


if __name__=='__main__':main()

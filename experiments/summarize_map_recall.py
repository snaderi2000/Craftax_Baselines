"""Summarize the overnight map pilots and route interventions as results arrive."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def fraction(values):
    return f'{100*np.mean(values):.1f}%' if len(values) else '—'


def interval(diff):
    # Rows are paired worlds. Average the four continuations or both seeds within each world first.
    rng=np.random.default_rng(71653)
    boot=np.concatenate([diff[rng.integers(0,len(diff),(100,len(diff)))].mean(1) for _ in range(40)])
    low,high=np.quantile(boot,[.025,.975])*100
    return f'{100*diff.mean():+.2f} pp (95% interval {low:+.2f} to {high:+.2f})'


def write_report(d):
    jobs=json.loads((d/'jobs.json').read_text())
    seeds=json.loads((d/'worlds.json').read_text())['world_seeds']
    report=['# Overnight map results','', 'The six pilots start from the same pretrained policy. Each final evaluation plays one episode on each of 1,024 fresh paired worlds, with temperature 1 and a 10,000-step cap.','',
      '| Readout | Training seed | Status | All 22 | Diamond | Ate plant | Death before 1k | Distant known diamond and pick → later mined |',
      '|---|---:|---|---:|---:|---:|---:|---:|']
    scores={};terminal=0
    for job in jobs:
        base=Path(job['study'])/'map_mlp'
        f=base/'evaluation_wallclock/episodes.csv'
        if not (base/'evaluation_wallclock/summary.json').exists():
            progress=base/'progress.json'
            state=json.loads(progress.read_text())['state'] if progress.exists() else 'validation or launch'
            report.append(f"| {job['mode']} | {job['seed']} | {state} | | | | | |")
            continue
        data=rows(f)
        if [int(r['world_seed']) for r in data]!=seeds:raise ValueError('Mismatched worlds')
        ok=np.array([r['all_22']=='True' for r in data],dtype=float)
        scores[job['mode'],job['seed']]=ok
        opportunities=[r for r in data if r['first_action_distant_diamond_with_pick']]
        mined=sum(int(r['collect_diamond']) for r in opportunities)
        death=[int(r['length'])<1000 and (r['lava']=='True' or r['nonlava_death']=='True') for r in data]
        report.append(f"| {job['mode']} | {job['seed']} | evaluated | {int(ok.sum())}/1024 | {fraction([int(r['collect_diamond']) for r in data])} | {fraction([int(r['eat_plant']) for r in data])} | {fraction(death)} | {mined}/{len(opportunities)} |")
        terminal+=1
    report += ['', '## Paired all-22 differences','']
    training_seeds=sorted({j['seed'] for j in jobs})
    for mode,baseline in [('locations','masked'),('events','locations'),('events','masked')]:
        if all((m,s) in scores for m in (mode,baseline) for s in training_seeds):
            diff=np.mean([scores[mode,s]-scores[baseline,s] for s in training_seeds],axis=0)
            report.append(f'- {mode} versus {baseline}, averaged over the two training seeds: {interval(diff)}.')
    report+=['','Intervals resample held-out worlds. Two training seeds give only a limited measure of training variability. The distant-diamond subset depends on the policy trajectory.','']
    src=d/'source_evaluation/summary.json'
    if src.exists():
        report.append(f"Frozen starting policy on these worlds: {json.loads(src.read_text())['all_22_success_pct']:.2f}% all 22.")
    probe=d/'route_probe'; completed=(probe/'complete.json').exists()
    report+=['','## Frozen 256-action route probe','', 'Each qualifying state branches into ordinary play and up to 128 actions of observed-map guidance. Both branches use four paired continuations. Arrival means adjacency to the selected tile; achievement means any diamond collection or plant harvest during the horizon.','']
    if completed:
        trials=rows(probe/'pairs.csv') if (probe/'pairs.csv').stat().st_size else []
        for kind,n in json.loads((probe/'screening.json').read_text()).items():
            report.append(f"**{kind.capitalize()}** — {n['routable']} routable out of {n['eligible']} eligible snapshots from {n['screened']} screened worlds.")
            for metric in ['adjacent_step','achievement_step','died']:
                by={}
                for r in trials:
                    if r['resource']!=kind:continue
                    by.setdefault((r['world_seed'],r['attempt']),{})[r['guided']]=(r[metric]=='True' if metric=='died' else int(r[metric])>=0)
                if not by:continue
                worlds={}
                for (world,_),v in by.items(): worlds.setdefault(world,[]).append(int(v['True'])-int(v['False']))
                change=interval(np.array([np.mean(x) for x in worlds.values()]))
                report.append(f"- {metric}: ordinary {fraction([v['False'] for v in by.values()])}, guided {fraction([v['True'] for v in by.values()])}, {change}.")
            report.append('')
    else:
        report.append('Screening or paired continuations in progress.')
    dest=d/'results.md'
    tmp=dest.with_suffix('.tmp');tmp.write_text('\n'.join(report)+'\n');tmp.replace(dest)
    return terminal==len(jobs) and completed and src.exists()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    while not write_report(args.directory) and args.wait:
        time.sleep(45)

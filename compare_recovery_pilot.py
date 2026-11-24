"""Score the paired source, ordinary PPO, and recovery PPO on held-out worlds."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent/'concept_mapping/runs'
PILOT = ROOT/'map_recovery_12b_pilot'


def load(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    flags = lambda field: np.array([row[field]=='True' for row in rows])
    all22 = flags('all_22')
    return ([int(row['world_seed']) for row in rows], dict(
        all_22=all22,early_death=(flags('lava')|flags('nonlava_death')) &
        np.array([int(row['length'])<1000 for row in rows]),both=all22 & flags('timeout')))


def main():
    paths={'source':PILOT/'source_evaluation/episodes.csv'}
    for arm in ('control','recovery'):
        paths[arm]=ROOT/f'lava_only_map_12b_{arm}_100m/map_mlp/evaluation_wallclock/episodes.csv'
    seeds=json.loads((PILOT/'heldout_worlds.json').read_text())['world_seeds']
    samples={}
    for name,path in paths.items():
        worlds,samples[name]=load(path)
        if worlds!=seeds or len(set(worlds))!=1024:raise ValueError(f'Unmatched evaluation: {path}')
    rng=np.random.default_rng(293)
    report={'episodes':len(seeds),'counts':{name:{m:int(v.sum()) for m,v in data.items()}
                                            for name,data in samples.items()},'paired_differences_pp':{}}
    for ref in ('source','control'):
        differences={}
        for m,a in samples['recovery'].items():
            delta=a.astype(int)-samples[ref][m].astype(int)
            interval=np.percentile(rng.choice(delta,(5000,len(delta))).mean(1)*100,[2.5,97.5])
            differences[m]=dict(estimate=100*delta.mean(),ci95=interval.tolist())
        report['paired_differences_pp']['recovery_minus_'+ref]=differences
    (PILOT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

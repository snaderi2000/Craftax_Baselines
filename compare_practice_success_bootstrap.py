"""Report normal-environment evaluation of both matched continuation arms."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent/'concept_mapping/runs'
OUT=ROOT/'practice_success_bootstrap_cap30_5m'


def main():
    if not all((OUT/f'seed{s}_{c}/normal_evaluation_plus5m/evaluation.json').exists()
               for s in (50,52) for c in ('control','bootstrap')):
        print('Comparison pending until all four runs complete.');return
    for budget in (1,5):
        rows=[]
        for s in (50,52):
            paths={'parent_cap30':ROOT/f'rare_four_reuse30_50m/seed{s}_practice/normal_evaluation_50m/evaluation.json'}
            paths.update({c:OUT/f'seed{s}_{c}/normal_evaluation_plus{budget}m/evaluation.json' for c in ('control','bootstrap')})
            for c,p in paths.items():
                try:d=json.loads(p.read_text())
                except json.JSONDecodeError:return
                rows.append(dict(seed=s,condition=c,**d))
        with (OUT/f'comparison_plus{budget}m.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        keys=['returned_episode_returns']+sorted(k for k in rows[0] if k.startswith('Achievements/'))
        means={c:{k:float(np.mean([r[k] for r in rows if r['condition']==c])) for k in keys} for c in paths}
        differences={s:{k:next(r[k] for r in rows if r['seed']==s and r['condition']=='bootstrap')-
                            next(r[k] for r in rows if r['seed']==s and r['condition']=='control') for k in keys} for s in (50,52)}
        (OUT/f'comparison_plus{budget}m.json').write_text(json.dumps(dict(means=means,per_run=rows,
            paired_seed_differences=differences,note='Two training seeds, 512 normal episodes per policy; descriptive screen, not a significance claim.'),indent=2)+'\n')
    print('Saved paired comparisons for +1M and +5M.')


if __name__=='__main__':main()

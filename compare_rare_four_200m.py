"""Three-seed cap30 evaluations; historical optimistic-reset controls are UNMATCHED."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent/'concept_mapping/runs/rare_four_reuse30_200m'


def main():
    for budget in (5,20,33,50,100,150,200):
        conditions={'cap30':'practice'}
        if budget in (50,100,200):conditions['historical_baseline_optimistic']='historical_baseline'
        paths=[(seed,condition,ROOT/f'seed{seed}_{suffix}/normal_evaluation_{budget}m/evaluation.json')
               for seed in (50,51,52) for condition,suffix in conditions.items()]
        if not all(p.exists() for _,_,p in paths):continue
        try:rows=[dict(seed=seed,condition=condition,**json.loads(p.read_text())) for seed,condition,p in paths]
        except json.JSONDecodeError:continue
        with (ROOT/f'comparison_{budget}m.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        keys=['returned_episode_returns']+sorted(k for k in rows[0] if k.startswith('Achievements/'))
        means={c:{k:float(np.mean([r[k] for r in rows if r['condition']==c])) for k in keys} for c in conditions}
        (ROOT/f'comparison_{budget}m.json').write_text(json.dumps(dict(means=means,per_run=rows,
            matched_controls=False,caveat='Historical baselines use optimistic resets; cap30 uses standard resets. Three training seeds, not pooled episodes, are independent replications.'),indent=2)+'\n')
    print('Updated available three-seed evaluation summaries; historical baselines are not matched controls.')


if __name__=='__main__':main()

import argparse
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent/'concept_mapping/runs'


def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['reuse30','backward'],required=True)
    args=p.parse_args(); name='rare_four_reuse30_50m' if args.variant=='reuse30' else 'backward_practice_50m'
    out=ROOT/name
    if not all((out/f'seed{s}_practice/normal_evaluation_50m/evaluation.json').exists() for s in (50,51,52,53)):
        print('Comparison pending until all four seeds finish.');return
    diagnostics={s:json.loads((out/f'seed{s}_practice/run_manifest.json').read_text())['archive_summary'] for s in (50,51,52,53)}
    for budget in (5,20,50):
        rows=[]
        for seed in (50,51,52,53):
            paths={'baseline':ROOT/f'id_practice_50m/seed{seed}_baseline',
                   'rare_four10':ROOT/f'rare_four_practice_50m/seed{seed}_practice',
                   args.variant:out/f'seed{seed}_practice'}
            for condition,path in paths.items():
                try:d=json.loads((path/f'normal_evaluation_{budget}m/evaluation.json').read_text())
                except json.JSONDecodeError:return
                rows.append(dict(seed=seed,condition=condition,**d))
        with (out/f'comparison_{budget}m.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        keys=['returned_episode_returns']+sorted(k for k in rows[0] if k.startswith('Achievements/'))
        means={c:{k:float(np.mean([r[k] for r in rows if r['condition']==c])) for k in keys} for c in paths}
        (out/f'comparison_{budget}m.json').write_text(json.dumps(dict(means=means,per_run=rows,
            final_practice_diagnostics=diagnostics),indent=2)+'\n')
        lines=[f'# {args.variant} at {budget}M','',
               'Four training seeds; 512 original-normal-environment frozen episodes per policy.', '',
               '| Metric | '+' | '.join(means)+' |','|---|'+'---:|'*len(means)]
        for k in keys:
            label='Mean total return' if k=='returned_episode_returns' else k.split('/',1)[1]+' (%)'
            lines.append('| '+label+' | '+' | '.join(f'{means[c][k]:.3f}' for c in means)+' |')
        lines+=['','Results are descriptive; training seeds, not evaluation episodes, are independent replications.']
        (out/f'results_{budget}m.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()

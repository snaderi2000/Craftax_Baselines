import csv
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent/'concept_mapping/runs'
OUT = ROOT/'rare_four_practice_50m'


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--full_episode',action='store_true')
    args=parser.parse_args()
    own='full_episode10' if args.full_episode else 'rare_four10'
    if args.full_episode:
        OUT=ROOT/'full_episode_practice_50m'
    if not all((OUT/f'seed{s}_practice/normal_evaluation_50m/evaluation.json').exists() for s in (50,51,52,53)):
        print('Waiting for all four rare-four evaluations before aggregate reporting.')
        return
    diagnostics={s:json.loads((OUT/f'seed{s}_practice/run_manifest.json').read_text())['archive_summary'] for s in (50,51,52,53)}
    for budget in (5,20,50):
        rows=[]
        for seed in (50,51,52,53):
            paths={'baseline':ROOT/f'id_practice_50m/seed{seed}_baseline',
                   'uniform25':ROOT/f'id_practice_50m/seed{seed}_practice',
                   own:OUT/f'seed{seed}_practice'}
            if args.full_episode and all((ROOT/f'rare_four_practice_50m/seed{s}_practice/normal_evaluation_50m/evaluation.json').exists() for s in (50,51,52,53)):
                paths['rare_four10']=ROOT/f'rare_four_practice_50m/seed{seed}_practice'
            for condition,path in paths.items():
                try:
                    d=json.loads((path/f'normal_evaluation_{budget}m/evaluation.json').read_text())
                except json.JSONDecodeError:
                    return
                assert d['episodes']==512 and d['evaluation_seed']==2050
                rows.append(dict(seed=seed,condition=condition,**d))
        with (OUT/f'comparison_{budget}m.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        keys=['returned_episode_returns']+sorted(k for k in rows[0] if k.startswith('Achievements/'))
        means={c:{k:float(np.mean([r[k] for r in rows if r['condition']==c])) for k in keys}
               for c in paths}
        diff=[next(r['returned_episode_returns'] for r in rows if r['seed']==s and r['condition']==own)-
              next(r['returned_episode_returns'] for r in rows if r['seed']==s and r['condition']=='baseline') for s in (50,51,52,53)]
        (OUT/f'comparison_{budget}m.json').write_text(json.dumps(dict(budget_millions=budget,means=means,
            return_differences_by_training_seed=diff,per_run=rows,final_practice_diagnostics=diagnostics,
            caveat='Full versus short rare-four changes continuation boundaries; versus uniform25 or baseline also changes practice selection/share/reuse.'),indent=2)+'\n')
        lines=[f'# {own} practice at {budget}M','',
               'Frozen policies; 512 normal-environment episodes per policy, four training seeds.', '',
               '| Metric | '+ ' | '.join(means)+' |','|---|'+'---:|'*len(means)]
        for k in keys:
            name='Mean total return' if k=='returned_episode_returns' else k.split('/',1)[1]+' (%)'
            lines.append(f'| {name} | '+ ' | '.join(f'{means[c][k]:.3f}' for c in means)+' |')
        lines+=['',own+' minus baseline return by seed: '+str(diff),'',
                'Practice amount, selection and snapshot-reuse limits all changed; results do not isolate one cause.']
        (OUT/f'results_{budget}m.md').write_text('\n'.join(lines)+'\n')
    print('Saved comparisons against baseline and prior uniform-ID practice.')


if __name__=='__main__':
    main()

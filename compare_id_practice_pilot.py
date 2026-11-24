"""Four-seed matched evaluation report; no mutation of training or W&B."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent/'concept_mapping/runs/id_practice_50m'


def main():
    if not all((ROOT/f'seed{s}_{c}/normal_evaluation_50m/evaluation.json').exists()
               for s in (50,51,52,53) for c in ('baseline','practice')):
        print('Four-seed comparison pending: not all evaluations have finished.')
        return
    for budget in (5,20,50):
        rows = []
        for seed in (50,51,52,53):
            for condition in ('baseline','practice'):
                try:
                    summary = json.loads((ROOT/f'seed{seed}_{condition}/normal_evaluation_{budget}m/evaluation.json').read_text())
                except json.JSONDecodeError:
                    return
                rows.append(dict(seed=seed,condition=condition,**summary))
        with (ROOT/f'comparison_{budget}m.csv').open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        metrics = ['returned_episode_returns'] + sorted(k for k in rows[0] if k.startswith('Achievements/'))
        means = {c:{k:float(np.mean([r[k] for r in rows if r['condition']==c])) for k in metrics}
                 for c in ('baseline','practice')}
        differences = {k:[next(r[k] for r in rows if r['seed']==s and r['condition']=='practice')-
                          next(r[k] for r in rows if r['seed']==s and r['condition']=='baseline') for s in (50,51,52,53)]
                       for k in metrics}
        report = dict(budget_millions=budget,training_seeds=[50,51,52,53],episodes_per_policy=512,
            evaluation_seed=2050,means=means,paired_training_seed_differences=differences,per_run=rows)
        (ROOT/f'comparison_{budget}m.json').write_text(json.dumps(report,indent=2)+'\n')
        lines = [f'# ID-organized practice: {budget}M checkpoint results','',
            'Frozen evaluation only in original normal Craftax, 512 episodes per policy, four matched training seeds.',
            '', '| Metric | Baseline | ID practice |','|---|---:|---:|']
        for k in metrics:
            name = 'Mean total return' if k=='returned_episode_returns' else k.split('/',1)[1]+' (%)'
            lines.append(f'| {name} | {means["baseline"][k]:.3f} | {means["practice"][k]:.3f} |')
        lines += ['', 'Practice-minus-baseline return differences by seed: '+str(differences['returned_episode_returns']),
            '', 'Training seeds are the independent replications. Episode counts do not establish four-seed statistical significance.']
        (ROOT/f'results_{budget}m.md').write_text('\n'.join(lines)+'\n')
    print('Saved four-seed comparisons at 5M, 20M and 50M.')


if __name__ == '__main__':
    main()

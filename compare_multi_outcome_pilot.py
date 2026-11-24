"""Write three-seed return and all-achievement comparison after all runs finish."""
import csv
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent / 'concept_mapping/runs'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['multi_outcome','targeted_outcome'],default='multi_outcome')
    variant = parser.parse_args().variant
    files = [ROOT / f'{variant}_aux_5m_seed{s}/comparison.json' for s in (50,51,52)]
    if not all(p.exists() for p in files):
        print('Three-seed report pending: one or more runs have not finished.')
        return
    try:
        results = [json.loads(p.read_text()) for p in files]
    except json.JSONDecodeError:
        print('Three-seed report pending: a comparison is still being written.')
        return
    out = ROOT / f'{variant}_aux_5m_replications'
    conditions = ('baseline','stone_aux',variant)
    returns = {name:float(np.mean([r['conditions'][name]['mean_total_return'] for r in results]))
               for name in conditions}
    achievements = results[0]['conditions']['baseline']['achievement_success_pct']
    rows = []
    for achievement in achievements:
        row = dict(achievement=achievement)
        for name in conditions:
            row[name+'_success_pct'] = float(np.mean([
                r['conditions'][name]['achievement_success_pct'][achievement] for r in results]))
        rows.append(row)
    summary = dict(training_seeds=[50,51,52],training_steps_per_policy=4997120,
        evaluation_episodes_per_policy=256,mean_total_return=returns,
        predictor_minus_baseline=returns[variant]-returns['baseline'],
        predictor_minus_stone=returns[variant]-returns['stone_aux'],
        achievement_success_pct=rows,per_seed=results)
    (out/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'achievement_comparison.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [f'# Three-seed {variant.replace("_","-")} predictor results','',
        'Frozen policies; 256 original-environment episodes per policy, evaluation seed 2050.',
        'Training seeds 50, 51 and 52 each receive 4,997,120 interactions.', '',
        '| Condition | Mean total episode return (average across seeds) |',
        '|---|---:|']
    lines += [f'| {name} | {returns[name]:.4f} |' for name in conditions]
    lines += ['', f'| Achievement | Baseline (%) | Stone predictor (%) | {variant.replace("_"," ")} (%) |',
              '|---|---:|---:|---:|']
    lines += [f"| {r['achievement']} | {r['baseline_success_pct']:.2f} | {r['stone_aux_success_pct']:.2f} | {r[variant+'_success_pct']:.2f} |" for r in rows]
    lines += ['', 'Success means earned at least once per episode. Three training seeds are a small study;',
              'differences are descriptive and not automatically evidence of statistical significance.']
    (out/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(returns,indent=2))


if __name__ == '__main__':
    main()

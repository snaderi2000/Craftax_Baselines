"""Wait for all six 200M runs, then collect same-budget, per-training-seed results."""
import csv
import json
from pathlib import Path
import time

OUT = Path(__file__).resolve().parent/'concept_mapping/runs/diamond_density_curriculum_200m'


def main():
    groups = [OUT/'replications_seed50', OUT/'replications_seed51_52']
    while True:
        progress = []
        for group in groups:
            path = group/'progress.json'
            if path.exists():
                try:
                    status = json.loads(path.read_text())
                except json.JSONDecodeError:
                    # Supervisors may be updating their progress files.
                    continue
                if status['failures']:
                    raise RuntimeError(f'{group}: {status["failures"]}')
                progress.append(status)
        if len(progress) == 2 and sum(len(p['completed']) for p in progress) == 6:
            break
        time.sleep(20)
    for budget, filename, evaluation in [(200, 'comparison_200m', 'normal_evaluation'),
                                         (100, 'comparison_100m', 'normal_evaluation_100m')]:
        results, rows = {}, []
        for seed in [50, 51, 52]:
            for condition in ['baseline', 'curriculum']:
                directory = (OUT/'replications_seed50'/condition if seed == 50 else
                             OUT/'replications_seed51_52'/f'seed{seed}_{condition}')
                summary = json.loads((directory/evaluation/'summary.json').read_text())
                summary.update(training_seed=seed, condition=condition)
                results[f'seed{seed}_{condition}'] = summary
                row = dict(training_seed=seed, condition=condition, checkpoint=summary['checkpoint'],
                           episodes=summary['episodes'], diamond_collections=summary['diamond_collections'],
                           overall_collection_rate=summary['diamond_collections']/summary['episodes'])
                for metric in ['visible_episode_conversion', 'immediate_episode_conversion']:
                    for k in ['successes', 'opportunities', 'rate']:
                        row[f'{metric}_{k}'] = summary[metric][k]
                rows.append(row)
        (OUT/f'{filename}.json').write_text(json.dumps(results, indent=2)+'\n')
        with (OUT/f'{filename}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'All six {budget}M checkpoint evaluations ready: {filename}.csv', flush=True)


if __name__ == '__main__':
    main()

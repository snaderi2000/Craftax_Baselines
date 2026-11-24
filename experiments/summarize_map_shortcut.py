"""Build the paired overnight V5 comparison from frozen evaluations."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from summarize_map_recall import interval


def read(path):
    return json.loads(path.read_text())


def summarize(root):
    worlds = read(root / 'worlds.json')['world_seeds']
    lines = ['# V5 fine-map shortcut', '',
             'Continuations of the final V5 checkpoint. Each frozen evaluation uses the same 1,024 new worlds, one categorical episode per world, temperature 1.', '',
             '| Model | Training seed | GPU | Status | Latest additional steps saved |',
             '| --- | ---: | ---: | --- | ---: |']
    evaluations, comparisons, seeds = [], {}, set()
    terminal = 0
    for study in sorted(root.glob('v5_*')):
        cfg = read(study / 'queue_request.json')
        gpu = cfg['architectures'][0]['train_gpu']
        name = 'shortcut' if cfg['ppo']['map_fine_shortcut'] else 'control'
        seed = cfg['seed']
        seeds.add(seed)
        start = int(Path(cfg['resume_from']).name)
        base = study / 'map_mlp'
        status = read(study / 'queue_progress.json')['state']
        added = 0
        if (base / 'progress.json').exists():
            progress = read(base / 'progress.json')
            status = progress['state']
            added = progress.get('latest_checkpoint', start) - start
        terminal += status in ('complete', 'failed', 'wall_clock_complete', 'complete_with_failures', 'preempted')
        lines.append(f'| {name} | {seed} | {gpu} | {status} | {added:,} |')
        for label in ('plus_150m', 'plus_200m', 'wallclock'):
            folder = base / f'evaluation_{label}'
            if not (folder / 'summary.json').exists():
                continue
            summary = read(folder / 'summary.json')
            with (folder / 'episodes.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            if [int(r['world_seed']) for r in rows] != worlds:
                raise ValueError(f'Unpaired worlds in {folder}')
            if summary['training_steps'] <= start:
                continue
            count = summary['training_steps'] - start
            ok = np.array([r['all_22'] == 'True' for r in rows], dtype=float)
            comparisons.setdefault((label, count), {})[name, seed] = ok
            memory = summary['diamond_memory']
            mined = memory['worlds_mined_an_eligible_tile']
            eligible = memory['worlds_with_eligible_offscreen_diamond']
            evaluations.append(f"| {name} | {seed} | {count:,} | {int(ok.sum())}/{len(rows)} | "
                               f"{sum(int(r['collect_diamond']) for r in rows)}/{len(rows)} | "
                               f"{memory['worlds_never_saw_diamond']} | {mined}/{eligible} |")
    lines += ['', '## Frozen evaluations', '',
              '| Model | Seed | Additional steps | All 22 | Collected diamond | Never saw diamond | Mined a remembered off-screen tile / eligible episodes |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |', *evaluations,
              '', 'An episode becomes eligible when the agent has an iron pickaxe and an uncollected remembered diamond outside its local view. The numerator counts first diamond collections at a tile that previously met those conditions. Each episode counts once. Opportunity frequency depends on the trajectory.', '']
    for (label, steps), scores in sorted(comparisons.items()):
        paired = sorted(s for s in seeds if ('shortcut', s) in scores and ('control', s) in scores)
        if paired:
            delta = np.mean([scores['shortcut', s] - scores['control', s] for s in paired], axis=0)
            lines.append(f'At +{steps:,} steps ({label}), the paired shortcut minus control difference for all 22 is {interval(delta)}. Matched training seeds {paired}.')
    lines += ['', 'Intervals resample worlds, averaging available matched training seeds within each world. Final checkpoints can have different interaction counts under the time cap.']
    tmp = root / 'results.tmp'
    tmp.write_text('\n'.join(lines) + '\n')
    tmp.replace(root / 'results.md')
    return terminal == 4


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    while not summarize(args.directory) and args.wait:
        time.sleep(45)

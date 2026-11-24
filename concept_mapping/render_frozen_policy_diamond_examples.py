"""Render reproducibly sampled diamond probes using saved checkpoint values.

Supports critic-only and ordinary joint-training results. Reuses
render_collectability_minimal_grids.py; no inference or new rollouts.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from concept_mapping.plot_collectability_emergence import BANKS, make_branches, read_rows
from concept_mapping.render_collectability_minimal_grids import render, CONDITIONS


ROOT = Path('concept_mapping/runs/cross_actor_critic_transfer_100b_10b/diamond_ordering')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=int, default=10027008)
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--results-dir', type=Path, default=ROOT)
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    source = args.results_dir / 'per_checkpoint' / f'{args.checkpoint}.csv'
    rows = read_rows(source)
    def full_success(row):
        return float(row['V_A']) > max(float(row['V_' + key]) for key in 'BCD')
    bank = BANKS['diamond']
    reference = read_rows(bank / 'per_state_results.csv')
    assert len(rows) == len(reference)
    for i, (row, ref) in enumerate(zip(rows, reference)):
        assert int(row['state_index']) == i
        assert all(row[key] == ref[key] for key in ('seed', 'episode_id', 'timestep'))
    if not 1 <= args.count <= len(rows):
        raise ValueError('--count must be between one and the bank size')
    indices = np.random.default_rng(args.seed).choice(len(rows), args.count, replace=False)
    selected = [rows[i] for i in indices]
    branches = make_branches('diamond', bank)
    observations = np.stack([branches[key][indices] for key in CONDITIONS['diamond']], axis=1)
    out = args.out_dir or args.results_dir / f'examples_checkpoint_{args.checkpoint}'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'per_state_results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    np.savez_compressed(out / 'counterfactual_observations.npz', observations=observations,
                        state_indices=indices)
    render(SimpleNamespace(task='diamond', source_dir=out, out_dir=out,
                           font='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                           font_size=25, block_size=7, display_scale=4,
                           examples_per_grid=1, gap=18))
    metadata = dict(checkpoint=args.checkpoint, source_results=str(source), state_bank=str(bank),
                    sample_seed=args.seed, selection='Uniform random without replacement; no outcome filtering.',
                    n_examples=args.count, state_indices=indices.tolist(),
                    checkpoint_ordering_success_pct=100 * sum(full_success(r) for r in rows)/len(rows),
                    panel_order=[['A: diamond + iron pickaxe', 'B: diamond, no iron pickaxe'],
                                 ['D: no diamond + iron pickaxe', 'C: no diamond, no iron pickaxe']],
                    values='Unmodified saved critic values at this checkpoint, not the final checkpoint.',
                    edits='All visible diamond replaced with stone in absent conditions; only iron pickaxe removed in no-tool conditions.')
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()

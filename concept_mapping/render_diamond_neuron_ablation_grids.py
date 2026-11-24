"""Render held-out 2x2 diamond states with values before/after neuron ablation.

Uses the saved intervention outputs and the existing minimal grid renderer.
No new inference, training, rollouts, or checkpoint edits.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin, PdfImagePlugin
from concept_mapping.plot_collectability_emergence import BANKS, make_branches, read_rows
from concept_mapping.render_collectability_minimal_grids import render, CONDITIONS


ROOT = Path('concept_mapping/runs/diamond_critic_circuit_10b')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT)
    parser.add_argument('--layer', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--seed', type=int, default=50)
    args = parser.parse_args()
    bank = BANKS['diamond']
    reference = read_rows(bank/'per_state_results.csv')
    with np.load(args.results_dir/'heldout_interventions.npz') as saved:
        heldout = saved['heldout_indices']
        before = saved['baseline_values']
        after = saved[f'layer{args.layer}_k{args.k}_ablated']
    assert before.shape == after.shape == (4, len(heldout))
    assert np.isfinite(before).all() and np.isfinite(after).all()
    if not 1 <= args.count <= len(heldout):
        raise ValueError('--count must be between one and the held-out population size')
    positions = np.random.default_rng(args.seed).choice(len(heldout), args.count, replace=False)
    selected = heldout[positions]
    branches = make_branches('diamond', bank)
    observations = np.stack([branches[key][selected] for key in CONDITIONS['diamond']], axis=1)
    ranking = read_rows(args.results_dir/'neuron_ranking.csv')
    neurons = [int(r['neuron']) for r in ranking if int(r['layer'])==args.layer and int(r['rank'])<=args.k]
    assert len(neurons) == args.k
    if args.layer == 3 and args.k == 4:
        assert neurons == [147, 198, 11, 439]
    rows = []
    for position, index in zip(positions, selected):
        ref = reference[index]
        row = dict(state_index=int(index), seed=ref['seed'], episode_id=ref['episode_id'], timestep=ref['timestep'])
        for j, key in enumerate('ABCD'):
            np.testing.assert_allclose(before[j, position], float(ref[f'V_{key}']), atol=1e-5, rtol=1e-5)
            row[f'V_{key}'] = float(before[j, position])
            row[f'V_{key}_ablated'] = float(after[j, position])
        row['interaction_before'] = row['V_A']-row['V_B']+row['V_C']-row['V_D']
        row['interaction_after'] = row['V_A_ablated']-row['V_B_ablated']+row['V_C_ablated']-row['V_D_ablated']
        rows.append(row)
    out = args.results_dir/f'counterfactual_grids_layer{args.layer}_top{args.k}'
    out.mkdir(parents=True, exist_ok=True)
    with (out/'per_state_results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(out/'counterfactual_observations.npz', observations=observations, state_indices=selected)
    render(SimpleNamespace(task='diamond', source_dir=out, out_dir=out,
        font='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', font_size=25,
        block_size=7, display_scale=4, examples_per_grid=1, gap=18, after_values_suffix='ablated'))
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 23)
    sheets = []
    for number, row in enumerate(rows, 1):
        name = out/f'example_{number:02d}.png'
        grid = Image.open(name).convert('RGB')
        sheet = Image.new('RGB', (grid.width, grid.height+45), 'white')
        sheet.paste(grid, (0, 0))
        label = f"interaction: {row['interaction_before']:.4f} → {row['interaction_after']:.4f}"
        ImageDraw.Draw(sheet).text((sheet.width/2, grid.height+8), label, anchor='mt', font=font, fill='black')
        sheet.save(name); sheets.append(sheet)
    sheets[0].save(out/'all_examples.pdf', save_all=True, append_images=sheets[1:])
    protocol = json.loads((args.results_dir/'protocol.json').read_text())
    metadata = dict(checkpoint=protocol['checkpoint'], run=protocol['run'], layer=args.layer, neurons=neurons,
        selection='Uniform random held-out states, without replacement or filtering by ablation outcome.',
        sample_seed=args.seed, state_indices=selected.tolist(), n_examples=args.count,
        ablation=protocol['ablation'], source_values=str(args.results_dir/'heldout_interventions.npz'),
        panel_order=[['A: diamond + iron pickaxe','B: diamond, no iron pickaxe'],
                     ['D: no diamond + iron pickaxe','C: neither']],
        before_after='Same observation shown once per condition, with both original and ablated critic values.',
        interaction='V_A - V_B + V_C - V_D', new_inference_or_training=False)
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(dict(output=str(out), neurons=neurons, examples=rows), indent=2))


if __name__ == '__main__':
    main()

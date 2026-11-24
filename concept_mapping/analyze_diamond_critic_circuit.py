"""Held-out attribution and activation-intervention study of diamond/tool value.

No policy training, new rollouts, or persistent checkpoint modifications.
Final-layer interaction attribution is exact; earlier-layer scores are local
gradient-based discovery heuristics, evaluated by actual downstream execution.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from concept_mapping.plot_collectability_emergence import (
    BANKS, make_branches, read_rows, _init_network, _load_wandb_config,
    _restore_train_state, make_craftax_env_from_name,
)
from concept_mapping.probe_frozen_policy_collectability import component_arrays, fingerprint, write_csv
from concept_mapping.analyze_water_health_interaction_10k import make_counterfactuals
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


RUN = 'wandb/run-20260630_214658-m0mw4end'
CHECKPOINT = 9999941632
OUT = Path('concept_mapping/runs/diamond_critic_circuit_10b')
CONTRAST = np.array([1., -1., 1., -1.])  # A - B + C - D
SIZES = (1, 4, 8, 16, 32, 64, 128, 512)


def bootstrap(data, draws, seed):
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, draws, 100):
        ix = rng.integers(0, len(data), size=(min(100, draws-start), len(data)))
        samples.append(data[ix].mean(axis=1))
    samples = np.concatenate(samples)
    ci = np.quantile(samples, [.025, .975], axis=0)
    ratios = samples[:, 1:4] / samples[:, :1]
    ratio_ci = np.quantile(ratios, [.025, .975], axis=0)
    return dict(mean_ci95=ci.tolist(), ratio_to_baseline_ci95=ratio_ci.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--random-draws', type=int, default=32)
    parser.add_argument('--control-states', type=int, default=512)
    parser.add_argument('--bootstrap-draws', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=50)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    start_time = time.monotonic()
    bank = BANKS['diamond']
    branches = make_branches('diamond', bank)
    reference = read_rows(bank / 'per_state_results.csv')
    n = len(reference)
    ids = [(int(r['seed']), int(r['episode_id'])) for r in reference]
    assert len(set(ids)) == n == 3483
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(n)
    discovery = np.sort(permutation[:int(.7*n)])
    heldout = np.sort(permutation[int(.7*n):])
    assert not set(discovery) & set(heldout)
    protocol = dict(run=RUN, checkpoint=CHECKPOINT, seed=args.seed, n_states=n,
                    discovery_n=len(discovery), heldout_n=len(heldout), sizes=SIZES,
                    random_draws=args.random_draws, bootstrap_draws=args.bootstrap_draws,
                    primary='Last hidden layer, top 32 neurons selected by mean absolute exact interaction contribution on discovery episodes.',
                    earlier_layers='Exploratory: rank by mean absolute interaction activation contrast times averaged local downstream gradient.',
                    generic_control='Rank by mean absolute local gradient times activation deviation from discovery mean, without an interaction contrast.',
                    ablation='Replace selected activations by the pooled discovery mean, identical for every branch and control input.',
                    tool_add='Same-state donor: A -> B and D -> C for selected coordinates; execute remaining critic layers.',
                    tool_remove='Same-state donor: B -> A and C -> D for selected coordinates; execute remaining critic layers.',
                    specificity_controls='Ripe vs unripe plant and visible water vs no water at health 2/drink 0, drawn from episodes absent from the entire diamond bank.',
                    no_new_training_or_rollouts=True)
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    write_csv(out / 'episode_split.csv', [dict(state_index=i, seed=ids[i][0], episode_id=ids[i][1],
               timestep=reference[i]['timestep'], split='discovery' if i in set(discovery) else 'heldout') for i in range(n)])
    print(json.dumps(protocol, indent=2), flush=True)
    cfg = _load_wandb_config(RUN)
    env = make_craftax_env_from_name(cfg['ENV_NAME'], auto_reset=False)
    network = _init_network(cfg, env, env.default_params)
    state, _ = _restore_train_state(RUN, cfg, network, env, env.default_params, CHECKPOINT)
    params = state.params['params']
    before_hash = fingerprint(component_arrays(state.params, range(4, 8)))

    @jax.jit
    def capture(obs):
        h, activations = obs, []
        for index in range(4, 7):
            p = params[f'Dense_{index}']
            h = jnp.tanh(h @ p['kernel'] + p['bias'])
            activations.append(h)
        p = params['Dense_7']
        return jnp.stack(activations), (h @ p['kernel'] + p['bias']).squeeze(-1)

    def capture_all(obs):
        hs, vs = [], []
        for begin in range(0, len(obs), 512):
            h, v = capture(jnp.asarray(obs[begin:begin+512]))
            hs.append(np.asarray(h)); vs.append(np.asarray(v))
        return np.concatenate(hs, axis=1), np.concatenate(vs).astype(np.float64)

    activation, baseline = [], []
    errors = {}
    for key in 'ABCD':
        h, v = capture_all(branches[key])
        expected = np.array([float(r['V_' + key]) for r in reference])
        errors[key] = float(np.max(np.abs(v-expected)))
        np.testing.assert_allclose(v, expected, rtol=1e-5, atol=1e-5)
        activation.append(h); baseline.append(v)
    activation = np.stack(activation, axis=1)  # layer, condition, state, neuron
    baseline = np.stack(baseline)
    baseline_i = CONTRAST @ baseline
    w = np.asarray(params['Dense_7']['kernel']).reshape(-1).astype(np.float64)
    q = np.einsum('c,lcnh->lnh', CONTRAST, activation.astype(np.float64))
    contributions = q[2] * w
    reconstruction_error = float(np.max(np.abs(contributions.sum(axis=1)-baseline_i)))
    assert reconstruction_error < 1e-5
    np.savez_compressed(out / 'exact_final_layer_contributions.npz', contributions=contributions,
                        interaction=baseline_i, discovery_indices=discovery, heldout_indices=heldout,
                        seed=np.array([x[0] for x in ids]), episode_id=np.array([x[1] for x in ids]))
    print(f'Critic replay errors {errors}; exact decomposition max error {reconstruction_error:.3g}', flush=True)

    # Specificity controls are episode-disjoint from every diamond probe.
    controls, control_validation = [], []
    all_diamond_ids = set(ids)
    plant_rows = read_rows(BANKS['plant'] / 'per_state_results.csv')
    eligible = [i for i, r in enumerate(plant_rows) if (int(r['seed']), int(r['episode_id'])) not in all_diamond_ids]
    plant_indices = rng.choice(eligible, min(args.control_states, len(eligible)), replace=False)
    plant = make_branches('plant', BANKS['plant'])
    controls.append(('plant', np.stack([plant['ripe'][plant_indices], plant['unripe'][plant_indices]])))
    control_validation.append(dict(task='plant', n=len(plant_indices), eligible=len(eligible),
                                   episode_ids=[(plant_rows[i]['seed'], plant_rows[i]['episode_id']) for i in plant_indices]))
    water_dir = Path('concept_mapping/runs/water_health_interaction_10b_10000episodes')
    with np.load(water_dir / 'base_states.npz') as data:
        water_ids = list(zip(data['seed'].tolist(), data['episode_id'].tolist()))
        eligible = [i for i, pair in enumerate(water_ids) if tuple(pair) not in all_diamond_ids]
        water_indices = rng.choice(eligible, min(args.control_states, len(eligible)), replace=False)
        water = make_counterfactuals(np.asarray(data['observations'][water_indices], dtype=np.float32))
    controls.append(('water', np.stack([water['h2_d0_water'], water['h2_d0_no_water']])))
    control_validation.append(dict(task='water', n=len(water_indices), eligible=len(eligible),
                                   episode_ids=[water_ids[i] for i in water_indices]))
    ctrl_h = []
    ctrl_v = []
    for name, obs in controls:
        h, v = capture_all(obs.reshape(-1, 1345))
        ctrl_h.append(h.reshape(3, 2, obs.shape[1], 512))
        ctrl_v.append(v.reshape(2, obs.shape[1]))
    assert controls[0][1].shape[1] == controls[1][1].shape[1]
    control_activations = np.concatenate(ctrl_h, axis=1)
    control_values = np.concatenate(ctrl_v, axis=0)
    n_control = control_values.shape[1]
    base_control_effects = np.stack([control_values[0]-control_values[1], control_values[2]-control_values[3]])
    assert np.all(base_control_effects.mean(axis=1) > 0)

    summary_rows, random_rows, neuron_rows, response_rows, primary = [], [], [], [], {}
    per_state_saved = dict(heldout_indices=heldout, baseline_values=baseline[:, heldout],
                           baseline_interaction=baseline_i[heldout])
    primary_per_state = None
    validation_tests = []
    rng_controls = np.random.default_rng(args.seed + 9001)
    for layer in range(3):
        def downstream(h):
            for index in range(5+layer, 7):
                p = params[f'Dense_{index}']
                h = jnp.tanh(h @ p['kernel'] + p['bias'])
            p = params['Dense_7']
            return (h @ p['kernel'] + p['bias']).squeeze(-1)
        downstream_jit = jax.jit(downstream)
        gradient_jit = jax.jit(jax.grad(lambda h: downstream(h).sum()))
        h_discovery = activation[layer][:, discovery]
        gradient = np.asarray(gradient_jit(jnp.asarray(h_discovery.reshape(-1, 512)))).reshape(4, len(discovery), 512)
        if layer == 2:
            local_contribution = contributions[discovery]
        else:
            local_contribution = q[layer, discovery] * gradient.mean(axis=0)
        score = np.mean(np.abs(local_contribution), axis=0)
        ranking = np.argsort(-score, kind='stable')
        mean_activation = h_discovery.mean(axis=(0, 1))
        generic_score = np.mean(np.abs(gradient * (h_discovery - mean_activation)), axis=(0, 1))
        generic_ranking = np.argsort(-generic_score, kind='stable')
        for rank, neuron in enumerate(ranking, 1):
            neuron_rows.append(dict(layer=layer+1, module=f'Dense_{layer+4}', neuron=int(neuron), rank=rank,
                discovery_mean_abs_interaction_contribution=float(score[neuron]),
                discovery_mean_signed_interaction_contribution=float(local_contribution[:, neuron].mean()),
                generic_value_score=float(generic_score[neuron]),
                output_weight=float(w[neuron]) if layer == 2 else '',
                attribution='exact' if layer == 2 else 'gradient proxy'))
            if rank <= 32:
                response_rows.append(dict(layer=layer+1, neuron=int(neuron), discovery_rank=rank,
                    output_weight=float(w[neuron]) if layer == 2 else '',
                    **{f'mean_h_{key}': float(activation[layer, j, heldout, neuron].mean()) for j, key in enumerate('ABCD')},
                    **{f'std_h_{key}': float(activation[layer, j, heldout, neuron].std()) for j, key in enumerate('ABCD')}))
        h = activation[layer][:, heldout]
        target = baseline[:, heldout]
        # Candidate masks contain no held-out information.
        @jax.jit
        def interventions(masks, factual, donors):
            mixed = factual[None] + masks[:, None, None, :] * (donors[None] - factual[None])
            return downstream(mixed.reshape(-1, 512)).reshape(masks.shape[0], factual.shape[0], factual.shape[1])

        def intervene(masks, factual, donors):
            outputs = []
            # Small fixed mask batches avoid a giant simultaneous activation tensor.
            for begin in range(0, len(masks), 4):
                chunk = masks[begin:begin+4]
                padded = np.pad(chunk, ((0, 4-len(chunk)), (0, 0)), mode='edge')
                outputs.append(np.asarray(interventions(jnp.asarray(padded), jnp.asarray(factual), jnp.asarray(donors)))[:len(chunk)])
            return np.concatenate(outputs).astype(np.float64)

        add_donor = h[[0, 0, 3, 3]]  # A->B, D->C
        remove_donor = h[[1, 1, 2, 2]]  # B->A, C->D
        ablate_donor = np.broadcast_to(mean_activation, h.shape)
        control_h = control_activations[layer]
        control_donor = np.broadcast_to(mean_activation, control_h.shape)
        checks = intervene(np.stack([np.zeros(512), np.ones(512)]).astype(np.float32), h, add_donor)
        np.testing.assert_allclose(checks[0], target, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(checks[1], target[[0, 0, 3, 3]], atol=1e-5, rtol=1e-5)
        validation_tests.append(dict(layer=layer+1, identity_patch_max_error=float(np.max(np.abs(checks[0]-target))),
                                     full_tool_patch_max_error=float(np.max(np.abs(checks[1]-target[[0, 0, 3, 3]])))))
        print(f'Layer {layer+1}: top32={ranking[:32].tolist()}', flush=True)
        for k in SIZES:
            random_n = args.random_draws if k < 512 else 1
            masks = np.zeros((2+random_n, 512), dtype=np.float32)
            masks[0, ranking[:k]] = 1
            masks[1, generic_ranking[:k]] = 1
            for draw in range(random_n):
                masks[draw+2, rng_controls.choice(512, k, replace=False)] = 1
            ablated = intervene(masks, h, ablate_donor)
            added = intervene(masks, h, add_donor)
            removed = intervene(masks, h, remove_donor)
            control_ablated = intervene(masks, control_h, control_donor)
            ablated_i = np.einsum('c,mcn->mn', CONTRAST, ablated)
            removed_i = np.einsum('c,mcn->mn', CONTRAST, removed)
            patch_present = added[:, 1] - target[1]
            patch_absent = added[:, 2] - target[2]
            selectivity = patch_present - patch_absent
            fidelity = .5*(np.abs(added[:, 1]-target[0]) + np.abs(added[:, 2]-target[3]))
            control_after = np.stack([control_ablated[:, 0]-control_ablated[:, 1],
                                      control_ablated[:, 2]-control_ablated[:, 3]], axis=1)
            ctrl_effect_mae = np.abs(control_after-base_control_effects[None]).mean(axis=2)
            control_v_mae = np.abs(control_ablated-control_values[None]).mean(axis=(1, 2))
            baseline_mean = float(baseline_i[heldout].mean())
            if k == 512:
                np.testing.assert_allclose(ablated_i, 0, atol=1e-5)
                np.testing.assert_allclose(removed_i, 0, atol=1e-5)
                np.testing.assert_allclose(selectivity[0], baseline_i[heldout], atol=1e-5, rtol=1e-5)
            for draw in range(len(masks)):
                row = dict(layer=layer+1, k=k, method='interaction' if draw==0 else 'generic_value' if draw==1 else 'random',
                           random_draw=draw-2 if draw>=2 else -1, heldout_n=len(heldout),
                           baseline_mean_interaction=baseline_mean,
                           ablated_mean_interaction=float(ablated_i[draw].mean()),
                           ablation_removed_fraction=float(1-ablated_i[draw].mean()/baseline_mean),
                           tool_patch_present_effect=float(patch_present[draw].mean()),
                           tool_patch_absent_effect=float(patch_absent[draw].mean()),
                           tool_patch_mean_selectivity=float(selectivity[draw].mean()),
                           tool_patch_recovered_fraction=float(selectivity[draw].mean()/baseline_mean),
                           reverse_patch_remaining_fraction=float(removed_i[draw].mean()/baseline_mean),
                           tool_patch_fidelity_mae=float(fidelity[draw].mean()),
                           ablated_diamond_value_mae=float(np.abs(ablated[draw]-target).mean()),
                           control_value_mae=float(control_v_mae[draw]),
                           plant_effect_mae=float(ctrl_effect_mae[draw, 0]),
                           water_effect_mae=float(ctrl_effect_mae[draw, 1]),
                           plant_mean_effect_after=float(control_after[draw, 0].mean()),
                           water_mean_effect_after=float(control_after[draw, 1].mean()))
                if draw < 2:
                    summary_rows.append(row)
                else:
                    random_rows.append(row)
            per_state_saved[f'layer{layer+1}_k{k}_ablated'] = ablated[0]
            per_state_saved[f'layer{layer+1}_k{k}_tool_added'] = added[0]
            per_state_saved[f'layer{layer+1}_k{k}_tool_removed'] = removed[0]
            if layer == 2:
                # Mean ablation of a linear readout must remove precisely the
                # selected terms of the exact interaction decomposition.
                expected_i = baseline_i[heldout] - contributions[heldout][:, ranking[:k]].sum(axis=1)
                np.testing.assert_allclose(ablated_i[0], expected_i, atol=1e-5, rtol=1e-5)
            if k == 32:
                data = np.column_stack([baseline_i[heldout], selectivity[0], ablated_i[0], removed_i[0], fidelity[0]])
                ci = bootstrap(data, args.bootstrap_draws, args.seed+layer)
                selected_row = summary_rows[-2]
                random_selected = random_rows[-random_n:]
                primary[f'layer{layer+1}'] = dict(neurons=ranking[:32].tolist(), metrics=selected_row,
                    bootstrap_columns=['baseline_interaction', 'patch_selectivity', 'ablated_interaction', 'reverse_patch_interaction', 'patch_fidelity_mae'],
                    bootstrap=ci,
                    random_median_patch_fraction=float(np.median([r['tool_patch_recovered_fraction'] for r in random_selected])),
                    random_patch_fraction_range=[float(min(r['tool_patch_recovered_fraction'] for r in random_selected)), float(max(r['tool_patch_recovered_fraction'] for r in random_selected))],
                    random_median_control_value_mae=float(np.median([r['control_value_mae'] for r in random_selected])))
                if layer == 2:
                    primary_per_state = [dict(state_index=int(ix), seed=ids[ix][0], episode_id=ids[ix][1],
                        baseline_interaction=float(baseline_i[ix]), patch_selectivity=float(selectivity[0, i]),
                        ablated_interaction=float(ablated_i[0, i]), reverse_patch_interaction=float(removed_i[0, i]),
                        **{f'V_{key}': float(target[j, i]) for j, key in enumerate('ABCD')},
                        **{f'V_{key}_ablated': float(ablated[0, j, i]) for j, key in enumerate('ABCD')},
                        V_B_tool_patched=float(added[0, 1, i]), V_C_tool_patched=float(added[0, 2, i]))
                        for i, ix in enumerate(heldout)]
            print(f'  k={k}: held-out patch recovered {100*summary_rows[-2]["tool_patch_recovered_fraction"]:.1f}%; '
                  f'control MAE {summary_rows[-2]["control_value_mae"]:.4f}', flush=True)
            write_csv(out / 'intervention_summary.csv', summary_rows)
            write_csv(out / 'random_controls.csv', random_rows)
        del downstream_jit, gradient_jit, interventions
    write_csv(out / 'neuron_ranking.csv', neuron_rows)
    write_csv(out / 'candidate_activation_responses.csv', response_rows)
    write_csv(out / 'primary_heldout_per_state.csv', primary_per_state)
    np.savez_compressed(out / 'heldout_interventions.npz', **per_state_saved)
    assert fingerprint(component_arrays(state.params, range(4, 8))) == before_hash
    report = dict(protocol=protocol, validation=dict(manual_forward_reference_max_error=errors,
                  exact_decomposition_max_error=reconstruction_error, intervention_endpoints=validation_tests,
                  checkpoint_critic_sha256=before_hash, parameters_unchanged=True),
                  baseline_heldout_mean_interaction=float(baseline_i[heldout].mean()),
                  baseline_heldout_positive_interaction_pct=float(100*(baseline_i[heldout]>0).mean()),
                  control_validation=control_validation,
                  original_control_mean_effects=dict(plant=float(base_control_effects[0].mean()), water=float(base_control_effects[1].mean())),
                  primary_top32=primary, elapsed_seconds=time.monotonic()-start_time,
                  limitations=['One checkpoint and training seed; held-out episodes are not independent training replications.',
                               'The network causal effect of an intervention does not establish simulator causal understanding.',
                               'Coordinate-level localization is basis-dependent; selecting large output-contribution units is not sufficient to identify a semantic concept.',
                               'Final-hidden-layer patching is linear and its recovery equals selected interaction contribution; earlier-layer nonlinear interventions and specificity controls are more diagnostic.',
                               'Mean ablations and mixed activation vectors may be outside the factual activation distribution.',
                               'Target selection was conditioned on first diamond collection; visible diamonds replaced by mineable stone.'])
    (out / 'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    plot_results(out, summary_rows, random_rows, contributions[discovery], contributions[heldout])
    plot_activations(out, response_rows)
    key_findings = summarize_top_four(out, report)
    lines = ['# Diamond–pickaxe critic circuit probe', '',
        f'Final regular-PPO checkpoint {CHECKPOINT:,}; {len(discovery):,} discovery and {len(heldout):,} held-out episodes. No policy training or new rollouts.', '',
        'The primary analysis selects 32 of the 512 final hidden units by mean absolute interaction contribution on discovery episodes. '
        'Earlier layers use a gradient-based ranking proxy. All rankings and replacement means use discovery data only. '
        'Held-out causal interventions execute the remaining nonlinear layers.', '',
        'For A=diamond+pickaxe, B=diamond/no pickaxe, C=neither, D=no diamond+pickaxe, interaction I=A−B+C−D. '
        'The exact final-hidden contribution of neuron j is output_weight[j]×(hA[j]−hB[j]+hC[j]−hD[j]).', '',
        '| Layer | Top-32 recovered interaction | Median random-32 recovery | Control value MAE |', '| --- | ---: | ---: | ---: |']
    for layer in range(1, 4):
        r = primary[f'layer{layer}']; m = r['metrics']
        lines.append(f'| {layer} | {100*m["tool_patch_recovered_fraction"]:.1f}% | {100*r["random_median_patch_fraction"]:.1f}% | {m["control_value_mae"]:.4f} |')
    lines.extend(['', *key_findings, '', 'Recovery is the patch effect with diamond minus the patch effect without diamond, divided by the held-out mean original interaction. '
        'It is signed and may exceed 100% when contributions cancel. No claim of feature specificity follows solely from a high recovery fraction.', '',
        'Plant and water controls are from episodes outside the entire diamond bank. Control MAE measures prediction change under the same mean-ablation mask. '
        'Plant/water effect errors measure change in paired resource contrasts, rather than only an additive value shift.', '',
        'Ten thousand paired episode-bootstrap resamples give descriptive 95% intervals conditional on the discovery-selected set. '
        'Random-mask ranges describe 32 random subsets, not training-seed confidence intervals.', '',
        '[Intervention plots](intervention_curves.pdf), [attribution concentration](attribution_concentration.pdf), '
        '[neuron rankings](neuron_ranking.csv), [held-out per-state primary results](primary_heldout_per_state.csv), '
        '[complete summary](summary.json), [pre-analysis protocol](protocol.json).', '',
        '[Top-neuron activation patterns](top_neuron_activation_patterns.pdf) show held-out mean activations '
        'in the four conditions, with output weights. The associated CSV includes the top 32 candidates in each layer.', '',
        *report['limitations'], '', 'Reproduce in jax-v100-cuda12:', '',
        '```bash', 'XLA_PYTHON_CLIENT_PREALLOCATE=false python concept_mapping/analyze_diamond_critic_circuit.py', '```', ''])
    (out / 'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(primary=primary, baseline_interaction=report['baseline_heldout_mean_interaction'], elapsed=report['elapsed_seconds']), indent=2), flush=True)


def summarize_top_four(out, report):
    rankings = read_rows(out/'neuron_ranking.csv')
    neurons = [int(r['neuron']) for r in rankings if r['layer']=='3' and int(r['rank'])<=4]
    with np.load(out/'exact_final_layer_contributions.npz') as data:
        held = data['heldout_indices']
        contribution = data['contributions'][held][:, neurons].sum(axis=1)
        interaction = data['interaction'][held]
    rng = np.random.default_rng(report['protocol']['seed'])
    ratios = []
    draws = report['protocol']['bootstrap_draws']
    for start in range(0,draws,100):
        ix = rng.integers(0,len(held),size=(min(100,draws-start),len(held)))
        ratios.extend((contribution[ix].mean(axis=1)/interaction[ix].mean(axis=1)).tolist())
    ci = np.quantile(ratios,[.025,.975])
    selected = next(r for r in read_rows(out/'intervention_summary.csv') if r['layer']=='3' and r['k']=='4' and r['method']=='interaction')
    random = [r for r in read_rows(out/'random_controls.csv') if r['layer']=='3' and r['k']=='4']
    original = [report['baseline_heldout_mean_interaction'], report['original_control_mean_effects']['plant'], report['original_control_mean_effects']['water']]
    after = [float(selected['ablated_mean_interaction']),float(selected['plant_mean_effect_after']),float(selected['water_mean_effect_after'])]
    details = dict(neurons=neurons, heldout_n=len(held), interaction_fraction=float(contribution.mean()/interaction.mean()),
                   interaction_fraction_95ci=ci.tolist(),
                   random_median_fraction=float(np.median([float(r['tool_patch_recovered_fraction']) for r in random])),
                   original_means=dict(zip(['diamond_pickaxe_interaction','plant_ripe_advantage','water_advantage'],original)),
                   after_ablation_means=dict(zip(['diamond_pickaxe_interaction','plant_ripe_advantage','water_advantage'],after)),
                   plant_advantage_fraction_removed=float(1-after[1]/original[1]),
                   note='Descriptive k=4 point from the prespecified neuron-count curve; primary set remains k=32. Intervals are conditional on the discovery-selected set, not across training seeds.')
    (out/'top_four_interpretation.json').write_text(json.dumps(details,indent=2)+'\n')
    fig,ax = plt.subplots(figsize=(6.7,4),constrained_layout=True)
    x = np.arange(3)
    ax.bar(x-.17,original,.34,label='Original critic',color='#2563eb')
    ax.bar(x+.17,after,.34,label='Top four neurons ablated',color='#dc2626')
    ax.set_xticks(x,['Diamond × pickaxe\ninteraction','Ripe-plant\nadvantage','Water\nadvantage'])
    ax.set(ylabel='Mean paired value contrast',title='The dominant neurons also support plant valuation')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(frameon=False,prop={'size':8,'weight':'bold'})
    for i in range(3):
        for offset,values in [(-.17,original),(.17,after)]:
            ax.text(i+offset,values[i]+.012,f'{values[i]:.3f}',ha='center',fontweight='bold',fontsize=9)
    fig.savefig(out/'top_four_specificity.pdf');fig.savefig(out/'top_four_specificity.png',dpi=250);plt.close(fig)
    return [f'The four highest-ranked final-layer neurons {neurons} account for **{100*details["interaction_fraction"]:.2f}%** '
            f'of the held-out mean interaction (conditional episode-bootstrap 95% CI {100*ci[0]:.2f}–{100*ci[1]:.2f}%). '
            'This is a descriptive point from the prespecified size curve; the primary set remains 32 neurons.', '',
            f'Ablating these four reduces the interaction from {original[0]:.3f} to {after[0]:.3f}, and '
            f'the ripe-plant advantage from {original[1]:.3f} to {after[1]:.3f} '
            f'({100*details["plant_advantage_fraction_removed"]:.1f}% reduction), while the mean water advantage '
            f'changes from {original[2]:.3f} to {after[2]:.3f}. Therefore these are not demonstrated diamond-only features. '
            'Shared collectability or immediate-achievement valuation is a possible explanation, not an established semantic label.', '',
            '[Specificity plot](top_four_specificity.pdf) and [top-four statistics](top_four_interpretation.json).']


def plot_results(out, rows, random_rows, discovery_c, heldout_c):
    plt.rcParams.update({'font.weight':'bold', 'axes.labelweight':'bold', 'axes.titleweight':'bold', 'font.size':10, 'pdf.fonttype':42})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    for layer, ax in enumerate(axes, 1):
        for method, label, color in [('interaction','Interaction-ranked','#2563eb'), ('generic_value','Generic value-ranked','#d97706')]:
            data = [r for r in rows if r['layer']==layer and r['method']==method and r['k']<512]
            ax.plot([r['k'] for r in data], [100*r['tool_patch_recovered_fraction'] for r in data], marker='o', color=color, label=label)
        ks = [k for k in SIZES if k<512]
        draws = [[100*r['tool_patch_recovered_fraction'] for r in random_rows if r['layer']==layer and r['k']==k] for k in ks]
        ax.plot(ks, [np.median(a) for a in draws], color='#6b7280', ls='--', label='Random median')
        ax.fill_between(ks, [min(a) for a in draws], [max(a) for a in draws], color='#6b7280', alpha=.15, label='Random min–max')
        ax.set(xscale='log', xlabel='Selected neurons (of 512)', ylabel='Interaction recovered (%)', title=f'Hidden layer {layer}')
        ax.axhline(100, color='black', lw=.7, alpha=.4); ax.grid(alpha=.2)
        ax.legend(frameon=False, prop={'size':7,'weight':'bold'})
    fig.savefig(out/'intervention_curves.pdf'); fig.savefig(out/'intervention_curves.png',dpi=250);plt.close(fig)
    ranking = np.argsort(-np.mean(np.abs(discovery_c), axis=0), kind='stable')
    fig, ax = plt.subplots(figsize=(6,4), constrained_layout=True)
    for c, label, color in [(discovery_c,'Discovery','#2563eb'),(heldout_c,'Held-out','#dc2626')]:
        signed = c.mean(axis=0)
        ax.plot(np.arange(1,513), 100*np.cumsum(signed[ranking])/signed.sum(), label=label, color=color)
    ax.set(xscale='log', xlabel='Neurons ranked using discovery data', ylabel='Signed mean interaction explained (%)', title='Final hidden-layer decomposition')
    ax.axhline(100,color='black',lw=.7,alpha=.4);ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.savefig(out/'attribution_concentration.pdf');fig.savefig(out/'attribution_concentration.png',dpi=250);plt.close(fig)


def plot_activations(out, rows):
    selected = [r for r in rows if r['layer']==3 and r['discovery_rank']<=4]
    values = np.array([[r[f'mean_h_{key}'] for key in 'ABDC'] for r in selected])
    fig, ax = plt.subplots(figsize=(7.3,3.8), constrained_layout=True)
    picture = ax.imshow(values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(4), ['Diamond\n+ pickaxe', 'Diamond\nno pickaxe', 'No diamond\n+ pickaxe', 'Neither'])
    ax.set_yticks(range(4), [f'Neuron {r["neuron"]}, w={r["output_weight"]:+.3f}' for r in selected])
    ax.set_title('Top four interaction contributors: held-out activations')
    for i in range(4):
        for j in range(4):
            ax.text(j,i,f'{values[i,j]:+.2f}',ha='center',va='center',color='white' if abs(values[i,j])>.6 else 'black',fontweight='bold')
    fig.colorbar(picture,ax=ax,label='Mean tanh activation')
    fig.savefig(out/'top_neuron_activation_patterns.pdf');fig.savefig(out/'top_neuron_activation_patterns.png',dpi=250);plt.close(fig)


if __name__ == '__main__':
    main()

"""Paired six-condition frozen-critic probe of first zombie encounters."""
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from concept_mapping.analyze_stone_wood_pickaxe_interaction_final import (
    evaluate_values, _init_network, _load_wandb_config, _restore_train_state,
    make_craftax_env_from_name)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('concept_mapping/runs')
OUT = ROOT / 'combat_sword_interaction_10b_10000episodes'
RUN = 'wandb/run-20260630_214658-m0mw4end'
PAIRS = [('iron', 'wood'), ('stone', 'wood'), ('iron', 'stone')]


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    obs, rows, targets = [], [], []
    exclusions = 0
    for seed, folder in [(50, 'master_symbolic_10b_5000episodes'), (51, 'master_symbolic_10b_5000episodes_seed51')]:
        base = ROOT / folder
        meta = json.loads((base / 'dataset_metadata.json').read_text())
        for path in sorted((base / 'episodes').glob('episode_*.npz')):
            with np.load(path) as d:
                direction = d['direction'].astype(int)
                rr = 3 + np.array([0, 0, -1, 1])[direction]
                cc = 4 + np.array([-1, 1, 0, 0])[direction]
                hit = (d['actions'] == 5) & (d['sleeping'] == 0) & (d['visible_mobs'][np.arange(len(rr)), rr, cc, 0] > 0)
                indices = np.flatnonzero(hit)
                if not len(indices):
                    exclusions += 1
                    continue
                t = int(indices[0])
                o = d['observations'][t].copy()
                target = int((rr[t] * 9 + cc[t]) * 21 + 17)
                assert o[target] == 1 and o[-1] == 0
                assert np.argmax(o[1339:1343]) == direction[t]
                inv = d['inventory'][t].astype(int)
                row = dict(seed=seed, episode_id=int(d['episode_id']), timestep=t,
                           target_row=int(rr[t]), target_col=int(cc[t]),
                           original_strongest_sword=('iron' if inv[11] else 'stone' if inv[10] else 'wood' if inv[9] else 'none'))
                row.update({f'inventory_{k}': int(v) for k, v in zip(meta['inventory_names'], inv)})
                row.update({k: int(v) for k, v in zip(['health', 'food', 'water', 'energy'], d['intrinsics'][t])})
                row.update({f'achievement_{k}': int(v) for k, v in zip(meta['achievement_names'], d['achievements'][t])})
                rows.append(row); obs.append(o); targets.append(target)
    factual = np.stack(obs)
    n = len(rows)
    assert n == 9898 and exclusions == 102
    assert len({(r['seed'], r['episode_id']) for r in rows}) == n
    np.savez_compressed(OUT / 'state_bank.npz', observations=factual, target_index=targets,
                        seed=[r['seed'] for r in rows], episode_id=[r['episode_id'] for r in rows], timestep=[r['timestep'] for r in rows])
    print(f'Bank validated: {n} states; {exclusions} episodes without attacks', flush=True)
    config = _load_wandb_config(RUN)
    env = make_craftax_env_from_name('Craftax-Classic-Symbolic-v1', auto_reset=False)
    net = _init_network(config, env, env.default_params)
    state, checkpoint = _restore_train_state(RUN, config, net, env, env.default_params, 9999941632)
    values = {'factual': evaluate_values(state.params, net, factual, 512)}
    # Independent unjitted, single-observation call mirrors factual rollout inference.
    reference = np.array([float(np.asarray(net.apply(state.params, o[None])[1]).reshape(-1)[0]) for o in factual[:32]])
    error = float(np.max(np.abs(reference - values['factual'][:32])))
    assert np.allclose(reference, values['factual'][:32], atol=1e-5, rtol=1e-5)
    examples = np.random.default_rng(50).choice(n, 5, replace=False)
    render_obs = {}
    for j, sword in enumerate(['wood', 'stone', 'iron']):
        present = factual.copy()
        present[:, 1332:1335] = 0
        present[:, 1332+j] = 0.1
        absent = present.copy()
        absent[np.arange(n), targets] = 0
        keep = np.ones(1345, bool); keep[1332:1335] = False
        assert np.array_equal(present[:, keep], factual[:, keep])
        assert np.all(np.count_nonzero(present != absent, axis=1) == 1)
        assert np.all(present[np.arange(n), targets] == 1)
        assert np.all(absent[np.arange(n), targets] == 0)
        assert np.all(np.rint(present[:, 1332:1335]*10) == np.eye(3, dtype=int)[j])
        for condition, observations in [('present', present), ('absent', absent)]:
            key = sword+'_'+condition
            values[key] = evaluate_values(state.params, net, observations, 512)
            render_obs[key] = observations[examples].copy()
    for i, row in enumerate(rows):
        row.update({'V_'+k: float(v[i]) for k, v in values.items()})
    summaries = []
    groups = {'all': np.ones(n, bool), 'resources_above_2': np.array([min(r['food'], r['water'], r['energy']) > 2 for r in rows])}
    groups.update({'original_'+s: np.array([r['original_strongest_sword'] == s for r in rows]) for s in ['none','wood','stone','iron']})
    for a, b in PAIRS:
        dp = (values[a+'_present'].astype(float) - values[b+'_present'])
        da = (values[a+'_absent'].astype(float) - values[b+'_absent'])
        delta = dp-da
        for i, row in enumerate(rows):
            row.update({f'{a}_vs_{b}_present': float(dp[i]), f'{a}_vs_{b}_absent': float(da[i]), f'{a}_vs_{b}_interaction': float(delta[i])})
        for group, mask in groups.items():
            x = delta[mask]; rng = np.random.default_rng(50); boot = []
            for _ in range(100):
                ix = rng.integers(0, len(x), (100, len(x)))
                boot.extend(x[ix].mean(axis=1))
            lo, hi = np.percentile(boot, [2.5,97.5])
            summaries.append(dict(group=group, comparison=a+'_vs_'+b, n=len(x),
                mean_present=float(dp[mask].mean()), mean_absent=float(da[mask].mean()),
                mean_interaction=float(x.mean()), median_interaction=float(np.median(x)),
                fraction_positive=float((x>0).mean()), ci_low=float(lo), ci_high=float(hi)))
    write_csv(OUT / 'per_state_results.csv', rows)
    write_csv(OUT / 'summary.csv', summaries)
    ordering = {}
    for group, mask in groups.items():
        ordering[group] = {'n': int(mask.sum()), 'iron_gt_stone_gt_wood': float(((values['iron_present'][mask]>values['stone_present'][mask]) & (values['stone_present'][mask]>values['wood_present'][mask])).mean())}
        for a,b in PAIRS:
            ordering[group][a+'_gt_'+b] = float((values[a+'_present'][mask]>values[b+'_present'][mask]).mean())
    (OUT / 'validation.json').write_text(json.dumps(dict(checkpoint=checkpoint, policy='feed-forward ActorCritic', states=n, excluded_no_attack=exclusions,
        missing_fields=['Individual zombie health/identity and latent resource counters are not stored; no such latent state is edited.'],
        full_health='Assumed from first successful attack per episode and default zombie spawn health 5.',
        edit_validation='All non-sword observation coordinates unchanged; removal changes exactly target zombie channel.',
        factual_reference_max_error=error, bootstrap_samples=10000, bootstrap_seed=50,
        unique_key='(seed, episode_id)', ordering=ordering), indent=2))
    fig, ax = plt.subplots(figsize=(7,4))
    primary = [s for s in summaries if s['group']=='all']
    xx=np.arange(3)
    ax.bar(xx-.18,[s['mean_present'] for s in primary],.36,label='Target present')
    ax.bar(xx+.18,[s['mean_absent'] for s in primary],.36,label='Target absent')
    ax.set_xticks(xx,[s['comparison'].replace('_',' ') for s in primary]); ax.axhline(0,color='black',lw=.7)
    ax.set_ylabel('Mean critic sword-upgrade effect');ax.legend();fig.tight_layout()
    fig.savefig(OUT/'upgrade_effects.pdf');fig.savefig(OUT/'upgrade_effects.png',dpi=180);plt.close(fig)
    from craftax.craftax_classic.constants import load_all_textures
    from concept_mapping.render_stone_wood_pickaxe_interaction_examples import labeled_frame
    from PIL import Image
    textures=load_all_textures(7); panel=load_all_textures(28)
    selection=[]
    for k,i in enumerate(examples):
        frames=[]
        for condition in ['present','absent']:
            for sword in ['wood','stone','iron']:
                key=sword+'_'+condition
                frames.append(labeled_frame(render_obs[key][k],key,f"seed {rows[i]['seed']} ep {rows[i]['episode_id']} t {rows[i]['timestep']}",float(values[key][i]),textures,panel,7,4))
        w,h=frames[0].size; sheet=Image.new('RGB',(3*w,2*h),'white')
        for j,f in enumerate(frames):sheet.paste(f,((j%3)*w,(j//3)*h))
        filename=f'example_{k+1}.png';sheet.save(OUT/filename)
        selection.append(dict(file=filename,seed=rows[i]['seed'],episode_id=rows[i]['episode_id'],timestep=rows[i]['timestep']))
    write_csv(OUT/'example_selection.csv',selection)
    print(json.dumps(primary,indent=2),flush=True)


if __name__ == '__main__':
    main()

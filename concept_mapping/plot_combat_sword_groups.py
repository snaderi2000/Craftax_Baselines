"""Primary sword interaction distributions and paired bootstrap group estimates."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / 'runs/combat_sword_interaction_10b_10000episodes'


def main():
    with (OUT / 'per_state_results.csv').open() as f:
        rows = list(csv.DictReader(f))
    cols = ['iron_vs_wood_present', 'iron_vs_wood_absent', 'iron_vs_wood_interaction']
    all_values = np.array([[float(r[k]) for k in cols] for r in rows])
    bins = np.linspace(all_values[:, 2].min(), all_values[:, 2].max(), 85)
    fig, axes = plt.subplots(5, 1, figsize=(9, 10), sharex=True, layout='constrained')
    output = []
    for ax, group, color in zip(axes, ['all','none','wood','stone','iron'], ['#666666','#bb5566','#dd9944','#4488aa','#228866']):
        mask = np.array([group == 'all' or r['original_strongest_sword'] == group for r in rows])
        a = all_values[mask]; x = a[:, 2]
        rng = np.random.default_rng(50); boot = []
        for _ in range(100):
            indices = rng.integers(0, len(a), (100, len(a)))
            boot.append(a[indices].mean(axis=1))
        ci = np.percentile(np.concatenate(boot), [2.5,97.5], axis=0)
        result = dict(group=group, n=len(x), fraction_positive=float((x>0).mean()), median_interaction=float(np.median(x)),
                      weighted_contribution_to_overall_mean=float(x.sum()/len(rows)))
        for j, name in enumerate(['present','absent','interaction']):
            result.update({f'mean_{name}':float(a[:,j].mean()),f'{name}_ci_low':float(ci[0,j]),f'{name}_ci_high':float(ci[1,j])})
        output.append(result)
        ax.hist(x, bins=bins, density=True, color=color, alpha=.65)
        ax.axvline(0,color='black',lw=.8)
        ax.axvline(x.mean(),color='#aa0033',lw=1.7,label='Mean')
        ax.axvline(np.median(x),color='#173f8a',ls='--',lw=1.7,label='Median')
        label = 'All states' if group=='all' else f'Original strongest sword: {group}'
        ax.set_title(f'{label}  |  n={len(x):,}  |  mean={x.mean():+.4f}, median={np.median(x):+.4f}  |  positive={100*(x>0).mean():.1f}%',fontsize=10,loc='left')
        ax.set_ylabel('Density'); ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=9)
    axes[-1].set_xlabel('Iron-vs-wood interaction: upgrade effect with target − upgrade effect without target')
    fig.suptitle('Combat-dependent valuation by original sword group\nFull observed range; shared bins; each panel normalized separately',fontsize=13)
    for extension in ['pdf','png']:
        fig.savefig(OUT/f'primary_interaction_by_original_sword.{extension}',dpi=180)
    with (OUT/'primary_group_details.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    for r in output:print(r)
    neg=all_values[:,2][all_values[:,2]<0];pos=all_values[:,2][all_values[:,2]>0]
    print('Negative mean:',neg.mean(),'positive mean:',pos.mean())


if __name__=='__main__':
    main()

# Reward transition explorer

An offline, interactive HTML atlas of label-free k-means over Craftax transition
differences. It reuses the existing 2,000-episode passive dataset from the frozen
9,999,941,632-step policy and selects **only scalar reward = +1** (absolute
tolerance 1e-6). No rollout collection or policy training is needed.

Open [`../runs/reward_transition_explorer/index.html`](../runs/reward_transition_explorer/index.html)
directly in a browser. It embeds 21 newly fitted +1-only experiments, diagnostic scores,
inspectable before/after states, and Craftax textures. It has no CDN dependencies.

Choose a representation, k (32/64/128), and an available seed. The filter is fixed
to +1 reward so health-only +0.1 transitions never enter fitting or inspection.
Click a cluster or a projection point. Filter examples by event and episode split;
browse examples closest to or farthest from the centroid, or semantic mismatches.
The composition heatmap and metrics use **all eligible held-out transitions**;
the image inspector and scatter use an explicitly labeled subset. Export scores
as JSON or currently filtered inspection examples as CSV.

## Build

From the repository root, using the existing environment:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python concept_mapping/reward_transition_explorer/build.py
```

`--source` points to an experiment directory with `summary.json`, `settings.json`,
`transition_bank.npz`, and `features.npz`. The builder refits k-means from scratch
on every eligible +1 reference transition and saves the models under the output
directory's `models/` folder. It does not reuse centroids fitted on all positive
rewards. Subsequent builds reuse these +1-only models.
`--out` changes the destination. `--sample-size` controls the random +1-reward
inspection sample; additional small-cluster examples are included without
achievement balancing. Every inspected example has +1 reward. The exporter expects Craftax-Classic
1345-dimensional observations. It validates episode-disjoint splits and never
modifies the source dataset.

## New k-means fits

Run this server on the machine containing the data, then open the printed URL:

```bash
/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python concept_mapping/reward_transition_explorer/serve.py
```

Default URL: `http://127.0.0.1:8765`. On a remote machine, forward port 8765
(for example through your editor or SSH). The server binds only to loopback.

Expand **Fit a new k-means model**. Choose k, seed, reference fit cap, and
iterations. New fits also use **only +1-reward transitions**. CPU Lloyd k-means uses k-means++ and the full normalized
representation. The default fit cap is 20,000 reference transitions; the cap
samples without labels. Assignment and diagnostics use every eligible reference
and held-out transition. Only reference data enter fitting. New models and exact
fit indices are saved under the output directory's `fits/` subdirectory.

New models are selected in the current browser session; re-requesting identical
parameters reuses the saved result. Fitting a previously visible combination
replaces that combination in the browser session. Original experiment files
stay untouched. The static HTML remains the original cached atlas after reload.
No GPU or extra dependencies are needed for new fits. Server RAM usage is several
GB when loaded; fitting large k and raw vectors may take a few minutes.

## Interpretation

- `critic_delta` / `actor_delta`: frozen hidden representation after minus before.
- `raw_observation_delta`: full symbolic observation after minus before.
- `action_inventory_delta`: action one-hot plus normalized inventory change.
- `random_actor_delta`: randomly initialized actor control.

Each clustered point represents an action's before/after change, not an isolated
pre-action state. All vectors are L2-normalized before Euclidean k-means. Scalar rewards only select
transitions; reward and achievement flags are excluded from centroid fitting.
Labels enter only after fitting, to annotate clusters and score event recovery.
Cached models use CPU k-means++, 25 iterations, k=32/64/128, and critic seeds
50/51/52 (other representations have seed 50). They fit all 27,778 eligible
reference transitions and evaluate all 11,960 held-out +1 transitions. New fits
can use a reference cap and different iteration count, so matching k and seed
alone does not promise identical centroids.

The reference-majority event mapping is evaluated on held-out episodes. Macro F1
averages supported achievements; NMI / ARI compare partitions; held-out purity
describes composition and is not a classification accuracy estimate. Rare event
support is visible in the per-event table. Unsupported scores are shown as dashes.

PCA is a 2D inspection view fitted on a reference-only subset, not the feature
space used by k-means. Distances shown are in the original normalized feature
space. Images reconstruct local symbolic observations, not hidden world state;
gold outlines mark differences in view coordinates, including shifts from
movement. The bank preserves the actual post-terminal observation.

Craftax's scalar reward is `new_achievement_count + 0.1 * health_change`.
There are 39,738 reward = +1 transitions in this batch; all unlock exactly one
new achievement. +0.1 health recovery is excluded. Strict +1 filtering also
excludes 694 genuine achievement transitions with simultaneous health changes
(for example +0.9 or +1.1 total reward). Selection uses the scalar reward,
not achievement IDs. The original source bank still contains other rewards and
uniform samples, but they never enter the rebuilt explorer's models or examples.
Scores describe this mature policy's selected passive transitions. They do not
establish causal discovery, generalization to early training, or practice gains.

## Verification

```bash
OPENBLAS_NUM_THREADS=2 /home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python -m unittest test_reward_transition_explorer
node --check concept_mapping/reward_transition_explorer/app.js
```

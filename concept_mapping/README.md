# Concept Mapping

Offline concept discovery for symbolic Craftax transitions.

Browse experiments in the [organized run index](runs/README.md), arranged by
month in Pacific time. Existing run paths
remain valid. Refresh the browser with
`python concept_mapping/organize_runs.py` from the repository root.

## Health-dependent water and food valuation

Using the frozen **9,999,941,632-step** checkpoint and the 10,000 evaluation
episodes collected with seeds **50 and 51**, we tested whether the critic
jointly represents health, internal need, and external resource availability.
For water, we selected the first eligible state per episode exactly five steps
before an observed drink-level increase and required visible water. For food,
we selected states five steps before a food increase produced by a `DO` action
targeting a cow and required a visible cow. This yielded **9,473 water states**
and **9,692 cow states**, with at most one state per episode.

Each state generated eight counterfactuals crossing health `{2, 9}`, the
corresponding internal level `{0, 9}`, and external resource presence/absence.
Absent water replaced every visible water tile with grass; absent cows cleared
every visible cow channel. All other observation features were preserved, and
no actions, rewards, time advancement, training, or new rollouts occurred.

For resource `R`, define
`Delta_R(h,r) = V(h,r,R+) - V(h,r,R-)`. Primary success requires:

```text
Delta_R(2,0) > max(0, Delta_R(2,9), Delta_R(9,0), Delta_R(9,9)).
```

This requires the external resource to have positive value and to be most
valuable when the agent is both depleted and low on health. We also measured
the three-way interaction:

```text
I_R = [Delta_R(2,0) - Delta_R(2,9)]
    - [Delta_R(9,0) - Delta_R(9,9)].
```

| Quantity | Water | Cow/food |
| --- | ---: | ---: |
| Eligible states | 9,473 | 9,692 |
| Primary success rate | **82.25%** [81.48, 83.01] | **83.24%** [82.51, 83.98] |
| `Delta_R(2,0) > 0` | 94.46% | 99.49% |
| Depletion moderation: `Delta_R(2,0) > Delta_R(2,9)` | 98.47% | 99.82% |
| Health moderation: `Delta_R(2,0) > Delta_R(9,0)` | 85.63% | 83.53% |
| Positive three-way interaction | 92.58% | 83.49% |
| Mean three-way interaction | 0.2303 [0.2262, 0.2343] | 0.0766 [0.0745, 0.0788] |

Intervals are 95% percentile intervals from 10,000 paired episode-bootstrap
resamples (seed 50). The results show that the critic usually assigns the
largest external-resource advantage to the jointly depleted, low-health
condition. They characterize selected states preceding observed recovery
events and do not establish optimal behavior in arbitrary states.

- [Appendix LaTeX](runs/resource_health_interaction_10b_10000episodes/appendix_summary.tex)
- [Water summary](runs/water_health_interaction_10b_10000episodes/summary.json)
- [Water per-state results](runs/water_health_interaction_10b_10000episodes/per_state_results.csv)
- [Water examples](runs/water_health_interaction_10b_10000episodes/random_success_examples_2x4/all_examples.pdf)
- [Food summary](runs/food_health_interaction_10b_10000episodes/summary.json)
- [Food per-state results](runs/food_health_interaction_10b_10000episodes/per_state_results.csv)
- [Food examples](runs/food_health_interaction_10b_10000episodes/random_success_examples_2x4/all_examples.pdf)

## Ripe-plant collectability

Using the frozen 9,999,941,632-step PPO checkpoint, we selected one pre-action
state immediately before first ripe-plant consumption from each of 1,883
eligible episodes in the 10,000-episode dataset (collection seeds 50 and 51).
We changed only the directly targeted `RIPE_PLANT` tile to `PLANT`, preserving
all other observation features. **The critic valued the ripe plant higher in
1,870/1,883 states (99.31%; 95% paired episode-bootstrap CI: 98.94–99.63%).**
The mean value difference was +0.652. Intervals use 10,000 bootstrap resamples
with seed 50. An unripe plant may ripen later, so this tests the value of
immediate edibility. [Results](runs/plant_collectability_10b_10000episodes_ripe_vs_unripe/summary.csv)
and [appendix LaTeX](runs/plant_collectability_10b_10000episodes_ripe_vs_unripe/appendix_setup.tex).

## Iron collectability: full-dataset results

Using the frozen 9,999,941,632-step checkpoint and 10,000 evaluation episodes
(collection seeds 50 and 51), we selected states immediately before first iron
collection. Excluding 3,281 states with enough wood and stone to recraft a stone
pickaxe left **6,567 states**. We crossed stone-pickaxe retention/removal with
iron presence/absence, replacing **all visible iron with stone** in absent
branches and preserving other features. No training or new rollouts occurred.

Success required `V(iron, pickaxe)` to exceed all three alternative values,
with no ordering among those alternatives. **5,757/6,567 states succeeded
(87.67%; 95% CI: 86.87–88.46%)**, using 10,000 paired episode-bootstrap
resamples (seed 50). The corresponding diamond result was **95.61%** on 3,483
eligible states. These are distinct selected populations; stone remains
mineable, so it is not a completely neutral control.

[Per-state results](runs/iron_collectability_10b_10000episodes_all_visible_stone/per_state_results.csv)
and [full summary](runs/iron_collectability_10b_10000episodes_all_visible_stone/summary.json).

## Diamond collectability: full-dataset counterfactual evaluation

Using the frozen PPO checkpoint at **9,999,941,632 training steps** from
`wandb/run-20260630_214658-m0mw4end`, we selected states from the 10,000 saved
evaluation episodes (5,000 each with collection seeds 50 and 51). Of 3,504
episodes with diamond collection, we retained one pre-action state immediately
before the first successful collection per episode. The agent was awake,
facing the target diamond, and possessed an iron pickaxe, with no prior diamond
achievement. We excluded 21 states whose inventory contained at least one
each of wood, stone, iron, and coal, leaving **3,483 states**. This conservative
replacement-tool exclusion uses material sufficiency regardless of access to
a crafting table and furnace.

Each state produces four observations: diamond present/absent crossed with
iron pickaxe retained/removed. Retained means the factual pickaxe count;
removed means setting only that inventory entry to zero. In diamond-absent
branches, **all visible diamond tiles are replaced with stone**. Every other
observation coordinate is unchanged, including other inventory, health,
resources, creatures, direction, and lighting. Achievement flags remain
unchanged as metadata (not separate critic inputs). No actions, reward awards,
time advancement, training, or new rollouts occur. The critic is feed-forward.

Define A = V(diamond, pickaxe), B = V(diamond, no pickaxe),
C = V(no diamond, no pickaxe), and D = V(no diamond, pickaxe).
The primary success criterion is **A > max(B, C, D)**, with no ordering imposed
among B, C, and D. This differs from the earlier, stricter pilot hypothesis
that additionally required D to be lowest.

**3,330/3,483 states satisfy the primary criterion (95.61%; 95% CI:
94.92–96.27%).** Pairwise rates are A > B: 95.95%, A > C: 97.13%, and
A > D: 100%. Confidence intervals use 10,000 paired episode-bootstrap
resamples, seed 50, retaining all four values as a unit. Requiring a margin
greater than 1e-5 leaves the primary rate unchanged. The earlier stricter
ordering succeeds in 93.63% of states and is secondary, not the primary result.

Edit checks passed; evaluations of the five existing pilot examples matched
within 4.18e-7. These results describe critic ordering on selected immediately
collectible-diamond states, not distant reachability, optimal valuation, or a
population guarantee. Stone preserves a blocked tile but is itself mineable
and is not a completely neutral control.

- [Per-state results](runs/diamond_collectability_10b_10000episodes_all_visible_stone/per_state_results.csv)
- [Summary and confidence intervals](runs/diamond_collectability_10b_10000episodes_all_visible_stone/summary.json)
- [Appendix text (LaTeX)](runs/diamond_collectability_10b_10000episodes_all_visible_stone/appendix_setup.tex)
- [Five pilot grids](runs/collectability_three_tasks_10b_pilot/diamond/five_examples.pdf)
- [Reproduction script](analyze_diamond_collectability_full.py)

The [four-comparison checkpoint analysis](runs/diamond_collectability_10b_10000episodes_all_visible_stone/four_comparisons_over_checkpoints/README.md)
evaluates the full ordering and three pairwise comparisons across all 29 saved
ordinary PPO checkpoints, with both actor and critic learning. On the same
3,483 states, final rates are 95.61% for the full ordering, 100% for diamond
presence with the iron pickaxe retained, 95.95% for iron-pickaxe retention with
diamond present, and 32.41% for diamond presence without an iron pickaxe.
[All four plots](runs/diamond_collectability_10b_10000episodes_all_visible_stone/four_comparisons_over_checkpoints/four_comparisons.pdf),
per-state values, numerical sensitivity checks, and checkpoint summaries are saved.
Reproduce with `python concept_mapping/analyze_joint_policy_diamond_comparisons.py`.

The [critic circuit probe](runs/diamond_critic_circuit_10b/README.md) decomposes
the final critic's diamond–pickaxe interaction and tests activation interventions
on 1,045 held-out episodes after selecting candidates on 2,438 discovery
episodes. Four final-hidden neurons (147, 198, 11, 439) account for 95.04% of
the held-out mean interaction (conditional episode-bootstrap 95% CI
94.64–95.45%). Ablating them also removes 89.5% of the ripe-plant value
advantage while leaving the mean water advantage nearly unchanged, so they
are not established diamond-only features. The prespecified primary top-32 set
accounts for 98.07%. Earlier-layer patches, same-size random controls,
generic-value rankings, validation, and raw per-state results are saved.
[Activation patterns](runs/diamond_critic_circuit_10b/top_neuron_activation_patterns.pdf)
and [specificity controls](runs/diamond_critic_circuit_10b/top_four_specificity.pdf).

The [individual-neuron and upstream-trace follow-up](runs/diamond_critic_circuit_10b/followup_neuron_specificity_and_trace/README.md)
keeps the same four neuron IDs fixed across diamond, iron, ripe-plant, and
water probes. It finds substantial diamond/plant overlap but little contribution
to iron or water contrasts. It also identifies a four-parent approximation to
neuron 198's incoming computation, preserving its AND-like sign pattern in
99.52% of held-out diamond states. This is an internal response reconstruction,
not a replacement for the whole critic or a policy-performance result.

## Highlight: early/late inventory-swap experiment

**Late inventory lowered the critic's predicted value in both observation
contexts in 89.06% of episode pairs** (8,906/10,000; 95% paired-bootstrap CI:
88.45–89.67%).

### Setup and success criterion

We used the frozen PPO checkpoint at **9,999,941,632 training steps** and both
5,000-episode batches (collection seeds **50 and 51**). For each episode with
`T` transitions, early and late pre-action observations were selected at
zero-based indices `floor(0.2*(T-1))` and `floor(0.8*(T-1))`. Selection times
were deterministic, not random. All four combinations of observation context
and inventory were evaluated:

| Context | Early inventory | Late inventory |
| --- | --- | --- |
| Early observation | `V_EE` (factual) | `V_EL` (swapped) |
| Late observation | `V_LE` (swapped) | `V_LL` (factual) |

Only the **12 inventory entries** were swapped. Health, food, water, energy,
terrain, creatures, direction, lighting, and sleeping status were preserved
within each context. No actions, rewards, time advancement, training, or new
rollouts were involved. Success requires **both `V_EL < V_EE` and
`V_LL < V_LE`**.

### Results

| Context | Fraction where late inventory lowers value | Mean late-minus-early inventory effect | Median effect |
| --- | ---: | ---: | ---: |
| Early observation | 93.15% | -2.4233 | -1.9626 |
| Late observation | 91.99% | -1.0172 | -0.8598 |
| Both inequalities jointly | **89.06%** | — | — |

Confidence intervals use **10,000 paired episode-bootstrap resamples**, seed
**50**, keeping all four values together. Excluding **100 identical-inventory
pairs**, joint success is **89.96%** (95% CI: 89.35–90.56%). Requiring each
decrease to exceed `1e-5` gives 89.05% overall. All observation-edit checks
passed, and single-observation versus batched inference agreed within 1.67e-6
on the validation subset.

These results support a consistent inventory-dependent lowering of predicted
remaining return. They do **not alone establish** that inventory dominates
other signals or that the critic understands the environment's finite reward
structure. Early/late denotes relative episode time rather than achievement
stage; later states had more achievements in 9,219 pairs. Swapped inventories
may also create unusual state combinations.

### Outputs and reproduction

- [Full experiment report](runs/early_late_inventory_swap_10b_10000episodes/README.md)
- [Per-episode metadata and four critic values](runs/early_late_inventory_swap_10b_10000episodes/per_episode_results.csv)
- [Summary and confidence intervals](runs/early_late_inventory_swap_10b_10000episodes/summary.csv)
- [15 unfiltered random example grids](runs/early_late_inventory_swap_10b_10000episodes/example_grids/fifteen_inventory_swap_examples.pdf)
- [Validation details](runs/early_late_inventory_swap_10b_10000episodes/validation.json)

From the repository root, using the existing `jax-v100-cuda12` environment:

```bash
python concept_mapping/analyze_early_late_inventory_swap.py
python concept_mapping/render_inventory_swap_examples.py
```

## Frozen 10B policy: 10,000-episode collection

### Dataset summary

Using the final policy checkpoint of the 10B training run, we collected 10,000
new episodes to construct an evaluation dataset for subsequent experiments.
The dataset contained 9,436,937 transitions, with a median episode length of
632 steps; only five episodes reached a 10,000-step cap. Crafty collected a
diamond in 35.0% of episodes. Overall, 69.0% of episodes ended in lava, rising
to 78.0% among episodes where diamond was collected, compared with 64.2%
among episodes without a diamond collected. Immediately before lava entry,
99.2% had food, water, or energy at two or below, despite 57.2% having full
health. Among episodes that collected a diamond and subsequently ended in
lava, the median interval between these events was 97.5 steps.

### Collection protocol, policy, and seeds

Both completed batches used the same frozen PPO policy for
`Craftax-Classic-Symbolic-v1`: checkpoint **9,999,941,632 training steps** from
`wandb/run-20260630_214658-m0mw4end` (weights under `files/policies/`).
This is the final checkpoint of that approximately 10B-step run, not the
separate ongoing 100B training run. No policy updates were performed during
collection. Actions were sampled from the policy's categorical distribution,
not selected by greedy argmax. The policy was trained with discount factor
**gamma = 0.99**.

We collected 5,000 episodes with seed **50** and an additional 5,000 with seed
**51**, preserving the original batch and its analyses. Each batch initializes
a JAX PRNG with its seed and splits episode-specific reset, environment-step,
and policy-sampling keys. Episodes start from fresh environment resets and
continue until environment termination or the **10,000-step cap**; collection
does not stop at diamond acquisition.

| Batch | Seed | Episodes | Directory under `concept_mapping/runs/` |
| --- | --- | --- | --- |
| Original (complete) | 50 | 5,000 | `master_symbolic_10b_5000episodes/` |
| Additional (complete) | 51 | 5,000 | `master_symbolic_10b_5000episodes_seed51/` |

Parallel collection used GPUs **1–7**, leaving GPU 0 reserved for training.
Use `(seed, episode_id)` as the unique episode key across batches: each batch
uses local IDs 0–4999. The new seed controls environment and categorical action
sampling. Sharding preserves each episode's random stream across restarts.
Do not overwrite the original files or treat either batch alone as 10,000 episodes.

Launch/resume the additional batch from the repository root:

```bash
/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python -u \
  concept_mapping/collect_master_10b_parallel.py --seed 51 \
  --out_dir concept_mapping/runs/master_symbolic_10b_5000episodes_seed51
```

Detached session: `collect_master_10b_seed51`. Each batch contains
`collection_progress.json`, `dataset_metadata.json`, `episodes/episode_*.npz`,
per-worker logs under `workers/`, and an `episode_manifest.csv` merged after
successful worker completion. Each NPZ stores all real pre-action observations,
actions, rewards, done flags, achievements, derived inventory/resource fields,
and the final post-action observation and achievements. Collection stops on
environment termination or the cap, not on diamond collection.

Both batches completed successfully, with 5,000 episode files and manifest
entries each. The original batch reached the cap in 2 episodes; its parallel
stage took 36 minutes 9 seconds. The additional batch reached the cap in 3
episodes and took 38 minutes 40 seconds.

### Measurement definitions and analysis scope

Episode length counts executed actions, including the final transition.
Diamond collection is identified from the terminal `COLLECT_DIAMOND`
achievement flag. Lava endings are identified from the lava tile at the
player's position in the terminal observation. Resource levels and full health
(9 points) are measured in the last pre-action state, before lava entry.
The resource-threshold statistic counts each lava-ending episode once when
any of food, water, or energy meets the threshold. The diamond-to-lava interval
counts actions after first diamond acquisition through the terminal action.

The summary above combines both batches: 3,504 episodes collected diamond,
2,732 of those ended in lava, and 4,172 of the 6,496 episodes without diamond
ended in lava. Among all 6,904 lava endings, 6,847 had a resource at two or
below and 3,951 had full health. These observations do not establish intent
or a causal reason for lava entry.

The earlier `depletion_return_analysis/` results still refer only to the
original seed-50 batch (5,000 episodes); they have not been recomputed on the
combined dataset.

The initial unit of discovery is a transition concept:

```text
g_t = (s_t, a_t, s_{t+1})
```

Later we can extend this to sequence concepts:

```text
G_t = (s_t, a_t, s_{t+1}, ..., s_{t+k})
```

## Hypothesis

Exploration should occur in concept space rather than raw state space. By
rewarding discovery of underrepresented latent transition concepts, agents
should collect more diverse experience and train better world models with fewer
environment interactions.

## Why Symbolic Observations Need Structure

Craftax-Classic symbolic observations are flat vectors of length 1345, but the
first part is structured map data:

```text
7 x 9 cells
each cell = block one-hot 37 + item one-hot 5 + creature one-hot 36 + light 1
cell width = 79
map width = 63 * 79 = 4977 for full Craftax, smaller for Classic if the env
            exposes the documented full cell schema
```

In practice, `Craftax-Classic-Symbolic-v1` reports a 1345-vector. Because local
Craftax versions may pack the classic subset differently, this package keeps the
observation layout explicit and validates dimensions before training.

The first embedding should not be a blind MLP forever. We want:

```text
phi(s)        state embedding
psi(s, a)     state-action embedding
nu(s,a,s')    transition concept embedding
```

For the first implementation:

```text
z       = phi(s)
z_next  = phi(s')
a_emb   = Embedding(a)
psi     = normalize(MLP([z, a_emb]))
nu      = normalize(MLP([z, a_emb, z_next - z]))
```

The residual `z_next - z` encourages transition concepts to represent what
changed, not only where the agent currently is.

## Offline Pipeline

1. Collect symbolic transitions from a trained or partially trained PPO policy.
2. Train a transition-concept encoder on `(obs, action, next_obs)`.
3. Embed a large transition set with `nu`.
4. Run k-means on the embeddings.
5. Inspect clusters by representative transitions, actions, rewards, deltas,
   and achievement info.
6. Freeze encoder and centroids for later PPO concept bonuses.

## Contrastive Sampling

Initial positives:

```text
same transition under two augmentations
psi(s_t, a_t) paired with nu(s_t, a_t, s_{t+1})
same pseudo-cluster after a first k-means pass
```

Initial negatives:

```text
transitions from different episodes
distant timesteps
different pseudo-clusters after bootstrapping
```

Avoid early hard negatives that may be false negatives:

```text
adjacent transitions in the same episode
same achievement delta
very similar state deltas
same action with similar local change
```

## First Loss

Use a stable, offline objective:

```text
L = L_info_nce(psi(s,a), nu(s,a,s'))
  + L_instance(nu(aug1(g)), nu(aug2(g)))
  + lambda_forward * ||predict(phi(s), a) - stopgrad(phi(s'))||^2
```

The forward auxiliary term keeps the representation tied to environment
dynamics, which should make clusters more useful for world-model data selection.

## Next Files

```text
symbolic_obs.py       symbolic observation layout helpers
models.py             Flax concept encoder modules
train_concepts.py     offline training entrypoint
fit_kmeans.py         embed transitions and fit concept centroids
inspect_concepts.py   summarize cluster contents
```

## Short-practice success bootstrapping: cap-30 continuation

Results and manifests: `runs/practice_success_bootstrap_cap30_5m/`.
Launcher: `run_practice_success_bootstrap.py`; comparison:
`compare_practice_success_bootstrap.py` (both at repository root).

Two paired training seeds, 50 and 52. Each restores its own completed cap-30
49,995,776-step policy and 1/5/20-step snapshot archive, including remaining
reuse quotas and normal-episode rarity history. Compare unchanged short cap-30
practice against `--id_practice_success_bootstrap`. The latter adds the detached
discounted critic value of the post-achievement, pre-reset observation to the
learning target only when success is not an actual environment termination.
Timeout bootstrapping stays enabled in both arms; real rewards, maps, inventory,
achievement flags, reset rules and GAE reset boundaries are unchanged. No
separate normal-only critic is introduced.

Four GPU jobs, two conditions per seed, 5M additional requested interactions
each (4,997,120 actual; final checkpoint 54,992,896). Both conditions restart
Adam identically and anneal LR 1e-4 over the continuation budget, rather than
reuse the exhausted 50M schedule. Live episodes/RNG restart, so this is not an
exact interrupted-run restoration. Practice phase counters restart; bank
insertion totals and rarity histories retain their parent values. No replay
quotas are renewed, and new source-episode serials avoid parent collisions.

W&B and policy checkpoints enabled. Frozen +1M/+5M checkpoints each play 512
original-normal-environment episodes, evaluation seed 2050. Report all 22
achievement rates, especially diamond and wake-up, plus total return against
the paired continuation control and original parent. This two-seed short
recovery screen does not establish a from-scratch benefit or statistical
significance. Existing experiments remain unchanged unless the new flags are
explicitly enabled. Tests: `test_practice_bootstrap.py`.

## Rare-four short cap-30 practice: 200M, three seeds

Output root: `runs/rare_four_reuse30_200m/`. Fresh training seeds 50, 51 and
52, on GPUs 0, 1 and 2. Launcher at repository root:
`run_rare_four_practice.py --max_reuse 30 --total_timesteps 200000000 --seed 50 --gpu 0`
(51/GPU 1, 52/GPU 2). No checkpoint, optimizer or snapshot bank is inherited
from the completed 50M runs. Initial LR 2e-4 anneals over the entire 200M budget.

Keep the cap-30 algorithm unchanged: 128 environments, 13 designated practice
slots, standard resets, 64-step PPO rollouts, four epochs, eight minibatches,
gamma .99, lambda .8, clip .2, value coefficient .5, entropy .01, gradient norm
1. Four rarest eligible achievements are chosen from the last 512 NORMAL
completed episodes, warm-up 512, refresh 128. Each ID retains 64 own-normal
pre-event snapshots at offsets 1/5/20 actions. Uniform selected-ID and snapshot
sampling, cap 30 per snapshot, no backward curriculum or graduation. Stop
practice at target success, true environment termination or 64 actions; only
artificial timeouts are bootstrapped. No success bootstrap, reward/environment
changes, auxiliary heads or external demonstrations. Every practice interaction
counts toward the requested 200M (199,999,488 actual).

Save the prior logarithmic checkpoints plus 50M, 75M, 100M and 150M, and the
final policy. Evaluate frozen 5M/20M/33M/50M/100M/150M/200M policies on 512
original-normal-environment episodes per seed, evaluation seed 2050, reporting
all 22 achievements and return. Evaluations run automatically after training.
W&B now retains practice/rarity diagnostics, and per-ID practice-interaction
counters are saved so allocation can be reported by steps, not just starts.
These logging changes do not alter learning or snapshot sampling.

Re-evaluate the existing 200M baseline seeds with the same evaluator at
50M/100M/200M, without retraining or changing their saved artifacts. Copies of
the new evaluation results live under `seed*_historical_baseline/` in this run
root. IMPORTANT: these historical baselines used optimistic resets, whereas
cap30 uses standard resets; they are not matched controls. No extra baseline
training has been launched. Aggregate CSV/JSON reports generated by
`compare_rare_four_200m.py` explicitly preserve that caveat. The 200M runs'
earlier checkpoints also differ from the original 50M experiments in their
learning-rate schedule, and should not be presented as the same training run.

## Offline achievement-event representation screen

Script: `test_achievement_representations_offline.py` (run from repository root
as `python concept_mapping/test_achievement_representations_offline.py`).
Outputs: `runs/achievement_representation_offline_10b_batch2000/`.
Tests: `test_offline_achievement_representations.py` at repository root.

Analysis seed 50 selects 1,000 complete episodes from each existing collection
seed, 50 and 51. Both collections used the frozen 9,999,941,632-step policy,
`wandb/run-20260630_214658-m0mw4end`. No new rollouts, PPO updates or encoder
training. Split episodes 70/30 within each collection: 1,400 reference episodes
and 600 held-out episodes. Extract single first-achievement transitions using
flag changes; actual terminal observations are retained, not auto-reset states.
There are 40,432 events, including 28 terminal events and no compound events.

Represent each event by its normalized post-minus-pre actor or critic hidden
vector. Retrieve the nearest event from a fixed balanced reference bank of
128 examples per achievement (2,816 total), scoring all 12,163 held-out events.
All 22 achievements have reference and test coverage. Controls use action,
inventory change, their combination, raw observation change, and an identically
structured random actor network. Average correctness across nearest-neighbor
ties rather than using arbitrary reference order. Achievement IDs identify
event boundaries, balance the reference bank and score retrieval; they never
train or enter the encoder. Thus this is labeled-reference retrieval, not
unsupervised discovery or an implementation of Achievement Distillation.

Held-out event-weighted retrieval accuracy: actor 96.61%, critic 98.79%, raw
observation 87.98%, random actor 86.78%, action plus inventory 83.08%, inventory
74.41%, action 50.91%. Critic 95% CI is [98.60%, 98.99%]; advantage over action
plus inventory is 15.71 percentage points [15.46, 15.95]. Use 2,000 paired
episode-bootstrap draws, conditional on the fixed reference bank and one
trained policy. Critic equal-achievement-weighted accuracy is 98.86%; diamond
is 100% on 208 held-out events, while place-stone is weakest at 86.98% on 599.

Saved episode selection, event observations, embeddings, per-event retrieval,
per-achievement scores, bootstrap summaries and PDF/PNG confusion plots make
the screen reproducible. Frozen hidden extraction agrees with the network's
actor probabilities and values within 1e-5, and reference/test episodes are
disjoint. Results support transferable event identity in the frozen features,
not causal prerequisites, hierarchy, future prediction or policy improvement.
This uses a mature 10B policy and does not establish representation quality in
the current 50M/200M practice runs; those require separate checkpoint tests.

## Label-free achievement clustering screen

Script: `test_achievement_clustering_offline.py`; tests at repository root:
`test_offline_achievement_clustering.py`. Outputs:
`runs/achievement_clustering_offline_10b_batch2000/`. Reuses the same 2,000
episodes and 1,400/600 reference/test episode split as the retrieval screen.
The encoder is the frozen 9,999,941,632-step PPO policy, not an AD-trained
encoder. No PPO training, auxiliary training, new rollouts or source edits.

Sampling does NOT use achievement flags: select every nonzero-reward
transition plus 64 uniformly sampled steps per episode (or all steps if fewer
than 64). Use actual post-terminal observations. The resulting union contains
200,563 transitions. Positive-reward clustering fits 42,487 reference events
and tests 19,429 held-out events, including health regeneration. Mixed clustering
fits 138,838 and tests 61,725 transitions, including negative health changes and
ordinary transitions. Mixed sampling enriches reward events; it is not the
natural per-step distribution. Uniform-subset diagnostics are also reported.

Fit Euclidean k-means on normalized transition differences, with k-means++
initialization on up to 8,192 unlabeled reference vectors and 25 Lloyd updates.
Predefined k sweep: 32, 64 (primary), 128; clustering seed 50 for five feature
methods, plus seeds 51/52 for the critic. Neither IDs nor held-out labels select
centroids, initialization, sampling or k. AFTER fitting, reference achievement
IDs name clusters by majority vote solely to score held-out precision, recall
and F1. There can be several clusters per achievement; 22 clusters are not
assumed. Health-only gain/loss, no-new-achievement and compound events are
separate diagnostic classes. Labels are never supplied to the policy.

Primary 64-cluster, seed-50 macro F1 across the 22 achievements:

| Feature | Positive-reward candidates | Mixed transitions |
| --- | ---: | ---: |
| Critic latent change | 91.54% | 64.88% |
| Actor latent change | 76.45% | 40.81% |
| Raw observation change | 62.33% | 36.60% |
| Random actor latent change | 68.28% | 33.87% |
| Action plus inventory change | 72.21% | 56.68% |

For critic seeds 50/51/52, positive-screen macro F1 is 91.54/91.09/97.07%,
and mixed-screen macro F1 is 64.88/67.23/69.40%. With 128 clusters, positive
scores are 97.37/97.25/97.33%; mixed scores 74.96/82.65/82.42%. These are
clustering initialization replications, NOT independent PPO training seeds.
The uniform-only mixed queries give lower critic 64-cluster macro F1,
56.27/56.19/57.27%, highlighting the effect of candidate enrichment.

Diamond illustrates the distinction between retrieval and unsupervised
grouping. At primary k=64/seed50, positive-screen diamond recall is 100% on
208 events but precision is 65.41%: its mapped group also contains 110 plant
events. Plant receives no separate majority-mapped group in this setting.
At k=128 diamond precision is 97.18--100% and recall 99.52% across critic
initializations. Mixed k=64 diamond precision is 69.23--69.80% and recall
99.52--100%, with residual contamination, particularly plant events.

Saved centroids/assignments, reference/test contingencies, per-achievement
scores, cluster compositions, summaries, selection and PDF/PNG plots permit
inspection. Results support useful reward-event geometry, not a complete
unfiltered event detector or a demonstrated online practice benefit. Ordinary
steps can include repeated physical achievements without a new reward/flag,
so the diagnostic first-unlock labels need not match physical event identity.
Symbolic observations expose inventory changes; controls address this shortcut
but do not establish causal prerequisites. Before online deployment, test earlier
PPO checkpoints and label-free cluster stability; importing a 10B encoder into
early training would be teacher transfer, not discovery by that early agent.

## AD-style transport grouping versus k-means (no auxiliary loss)

Script: `compare_transport_achievement_groups.py`; tests at repository root:
`test_transport_achievement_groups.py`. Outputs:
`runs/transport_vs_kmeans_achievement_groups_10b_batch2000/`.
Reuse the 2,000 complete episodes and their disjoint 1,400/600 reference/test
split. Frozen checkpoint 9,999,941,632, collection seeds 50/51. No PPO updates,
auxiliary encoder learning or new rollouts. This is NOT a full AD experiment.

Detect achievement-sized reward events without flags by subtracting the known
Classic health reward (.1 per changed health point) from the net reward and
thresholding the residual at .5. This corrects simultaneous health damage or
regeneration; requiring net reward exactly +1 would miss valid achievements.
The resulting 40,432 events contain 28,269 reference and 12,163 held-out
transitions. Post-selection flag audit finds zero spurious/missed achievement
events. Health-reward-only transitions are excluded, unlike the earlier
positive-reward clustering screen; its scores are not directly comparable.

Use the same frozen normalized critic or actor latent changes for both methods.
AD-style matching follows the official upstream rule: unit event masses,
cosine cost, POT `entropic_partial_wasserstein`, regularization .05, 100
iterations, and retain transport entries above .5. Existing POT 0.9.1 is reused
from `ad-crafter`, without installing or modifying a conda environment.
Source: https://github.com/snu-mllab/Achievement-Distillation/blob/main/achievement_distillation/algorithm/ppo_ad.py

AD itself does not define persistent clusters. Our grouping extension randomly
chooses a reference anchor among the 115 episodes with the greatest number of
reward events (22, inferred from data). Match each reference episode to it;
one group per anchor event. The finite 100-iteration solver occasionally leaves
capacity violations and conflicting above-.5 edges. Preserve raw edges for
faithful matching diagnostics; greedily accept edges in descending confidence
under one-to-one constraints ONLY for the persistent-group adapter. Conflicts
and capacity residuals are recorded, not silently hidden. Average confidently
assigned reference event vectors to form numerical group prototypes.

Compare k-means with 22, 32, 64 and 128 groups on exactly the same vectors and
reference events, using 25 Lloyd updates and k-means++ initialization. K=22
explicitly supplies a concept-count prior but no semantic labels. Repeat both
grouping methods with seeds 50/51/52; these are grouping/anchor initializations,
not separate PPO training seeds. Both methods classify each new held-out event
independently by nearest Euclidean prototype in normalized feature space.
Separately report whole-episode transport assignment, which has access to an
entire episode's event set and is not an online single-event classifier.

Ground-truth IDs are used ONLY after fitting to name groups by reference
majority and score held-out events. They never enter discovery, centroid
fitting, anchor selection, inference models or rarity counters.

Mean held-out macro F1 across grouping seeds, averaging the 22 achievements:

| Grouping | Groups | Critic features | Actor features |
| --- | ---: | ---: | ---: |
| Transport anchor plus prototypes | 22 | 96.78% | 77.35% |
| K-means | 22 | 79.76% | 62.27% |
| K-means | 32 | 88.24% | 72.55% |
| K-means | 64 | 98.25% | 85.33% |
| K-means | 128 | 98.45% | 92.37% |

Critic transport groups cover all 22 semantic achievements in each run;
22-group k-means covers 19/19/20. For primary seed50, transport prototype
accuracy is 97.43%, macro F1 97.29% [96.94,97.62] under 2,000 paired
episode-bootstrap draws conditional on the fitted reference bank. Diamond
precision is 96.30%, recall 100% (208 held-out events). Thus transport supplies
a compact 22-group bank, while overclustered k-means gives slightly higher
classification F1 at the expense of more group fragmentation. These results
do not establish an improvement in PPO learning or early-training discovery.

The 32-group extension used `--resume`, fitting only six new models (critic
and actor features, seeds 50/51/52). Existing fitted models and per-event
artifacts are left untouched; all 30 previous aggregate result rows were
verified unchanged. The previous JSON summary is preserved as
`summary_before_kmeans32.json`. Updated plots, summaries, group frequencies
and lead-up indices include the new condition. Critic k=32 accuracy is 92.98%
and macro F1 88.24%; individual F1 values are 87.86/85.56/91.30%. Reference
majority naming covers 21/20/21 achievements; eat-plant lacks a separately
named group in all three runs, and collect-coal also does in seed51. Thus
32 groups improves over 22-group k-means but does not reliably isolate all
22 achievements. The primary seed50 macro-F1 episode-bootstrap interval is
[87.49%,88.23%], conditional on that fitted grouping.

`*_model.npz` files contain prototypes, frozen encoder provenance and numeric
group IDs, but NO achievement names or semantic mappings. Group audit CSVs
contain names solely for evaluation. `group_frequency.csv` and
`grouped_reference_events.csv` provide normal-reference episode frequencies
and 1/5/20-step lead-up indices for later practice integration. Lead-ups are
relative to the post-event state; -1 means unavailable, not a clipped repeat.
These are event indices, not newly collected full simulator snapshots.

Reusable inference accepts an NPZ with `before[N,1345]`, `after[N,1345]` and
`rewards[N]`; no labels or actions are required for these actor/critic encoders.
The output is a numerical group ID per achievement event, or -1 for non-events:

```bash
python concept_mapping/compare_transport_achievement_groups.py \
  --predict_model concept_mapping/runs/transport_vs_kmeans_achievement_groups_10b_batch2000/critic_delta_transport_anchor_seed50_model.npz \
  --predict_input new_transitions.npz --predict_output new_groups.csv
```

Smoke verification: 64 held-out raw-observation events reproduce cached group
assignments, 16 ordinary transitions are rejected, and saved models contain no
semantic labels. A static observation by itself cannot identify an already-
completed event; predicting future goals remains a different task. The frozen
10B encoder is a mature teacher if reused in early PPO training, not evidence
that a fresh agent already discovers these concepts.

## Frozen 50M baseline encoder: fixed-probe grouping check

Outputs: `runs/transport_vs_kmeans_encoder50m_seed50_fixed10b_batch2000/`.
Use the ordinary, standard-reset PPO baseline trained with seed50, run
`wandb/run-20261013_011547-gtzkxoth`, final checkpoint 49,995,776. It received
no deliberate practice. This is an independently trained 50M-budget policy,
not the 50M milestone of the 10B run. Architectures match, but training budgets,
annealing schedules and environment batch sizes differ (128 versus 1,024).

Reuse exactly the 40,432 achievement-reward transitions from the original
10B-policy dataset (1,400 reference and 600 held-out episodes). Replace the
encoder weights and freshly embed both sides of every event; do NOT reuse the
10B actor/critic features. Encoder provenance, validation, feature cache and
inference model provenance are separate from collection-policy provenance.
Ground-truth labels remain evaluation-only. No training or new rollouts.

```bash
CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false \
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python \
  concept_mapping/compare_transport_achievement_groups.py \
  --encoder_run /home/shawheen/Craftax_Baselines/wandb/run-20261013_011547-gtzkxoth \
  --encoder_checkpoint 49995776 \
  --out concept_mapping/runs/transport_vs_kmeans_encoder50m_seed50_fixed10b_batch2000
```

Same POT/prototype and k-means protocol, grouping seeds 50/51/52. Mean
single-event macro F1 across the 22 achievements:

| Grouping | 50M critic | 10B critic, same probes | 50M actor |
| --- | ---: | ---: | ---: |
| Transport plus prototypes (22 groups) | 92.88% | 96.78% | 67.26% |
| K-means, 22 groups | 74.56% | 79.76% | 57.47% |
| K-means, 32 groups | 89.84% | 88.24% | 67.36% |
| K-means, 64 groups | 93.23% | 98.25% | 80.58% |
| K-means, 128 groups | 96.17% | 98.45% | 88.87% |

50M critic transport groups represent all 22 achievements in all three runs;
32-group k-means represents 21 in each. Primary transport seed50 macro F1 is
92.49%, conditional 95% episode-bootstrap CI [91.98%,92.97%]; accuracy is
93.07%. Diamond precision/recall are 81.40%/94.71%, plant 78.00%/88.64%.
Across transport seeds, the single-event prototype classifier ranks plant
eating and diamond as the two rarest groups in normal reference episodes.
Other rare-group ranks are less reliable (zombie/iron-tool/placement confusion).
This is supportive evidence for early critic event geometry, not proof of
correct rarity allocation or a benefit to learning.

`encoder_checkpoint_comparison.csv` summarizes the fixed-probe comparisons;
`prototype_rarity_audit.csv` audits deployable prototype-assignment frequencies
(not just accepted offline transport matches). Group names in that audit are
diagnostic only. Models remain numeric/label-free. Raw-observation inference
smoke test reproduces 64 cached assignments and rejects 16 ordinary transitions.
Manual hidden extraction agrees with actor probabilities within 2.98e-8 and
critic values exactly. The 50M critic features differ from 10B features (first
256 events' raw-feature RMS difference 0.4194), confirming distinct weights.

Critical limitation: the probes AND the 22-event reference anchors were
collected by the mature 10B policy. This does not establish that the 50M policy
has discovered all 22 events in its own experience. An own-trajectory event-bank
test is still needed before claiming fully online discovery or practice gains.
The script's `--encoder_run`/`--encoder_checkpoint` options support future
checkpoint tests and reject resumes with mismatched encoder/cache provenance.

## Own-policy 50M achievement-event grouping (no lead-ups)

Outputs: `runs/transport_own_policy_50m_seed50_1000episodes/`.
Script: `group_own_policy_achievement_events.py`.
Collect 1,000 fresh normal episodes from the same frozen standard-reset PPO
baseline, run `wandb/run-20261013_011547-gtzkxoth`, checkpoint 49,995,776
(training seed 50). Collection seed 3050; stochastic actions; 32 workers;
10,000-step cap. Collection took approximately 64 seconds and yielded 346,459
transitions, 17,377 achievement events, median episode length 356, and no capped
episodes. Save only event before/after observations, action, net reward,
episode/timestep/event IDs, terminal indicator and split. No full trajectories,
simulator snapshots, lead-ups, practice resets, auxiliary losses or training.

Event selection uses `reward - 0.1 * health_change > 0.5`, without achievement
IDs. Every saved transition had exactly one unit of achievement reward; 284
had simultaneous health rewards and therefore net reward other than +1.
True terminal successor observations are retained (10 terminal events), never
reset observations. Achievement flags/labels are saved separately in
`evaluation_labels.npz` and read only after all group fits are completed.

Split whole episodes 700/300 using seed 4050: 12,138 reference and 5,239
held-out events, with no episode overlap. Both collection and encoding use the
50M policy, with no 10B probes or anchor episodes. Apply the existing AD-style
POT matching plus our persistent prototype decoder to frozen actor/critic
after-minus-before hidden features. A maximum-event reference episode supplies
group IDs, without labels or a requested concept count. Grouping seeds 50/51/52
each yielded 20 groups. This is matching with a prototype extension, not full
Achievement Distillation training.

| Representation | Held-out accuracy | Macro F1 | Achievements represented across grouping seeds |
| --- | ---: | ---: | --- |
| Critic | 88.21% | 81.47% | 20 / 20 / 19 |
| Actor | 76.66% | 67.73% | 19 / 17 / 18 |

Accuracy and F1 are means over the three grouping seeds, not independent PPO
training seeds. Macro F1 equally weights the 21 achievement types observed in
the held-out data. The 20 reference types exclude plant eating and diamond:
there were no diamond events anywhere, and the sole plant-eating event fell
in the held-out split. These missing discoveries cannot be recovered by
forcing 22 group IDs. Primary critic grouping seed 50 achieved 88.68% accuracy
(95% episode-bootstrap CI 87.84–89.52%) and 82.41% macro F1 (81.59–83.22%);
2,000 bootstrap draws, conditional on the fitted grouping. The bootstrap uses
the same fixed supported-class definition as the reported macro F1.

Common-event classification is encouraging, but rarity targeting is not yet
robust. Primary critic F1 was 100% for wood collection, 99.25% for coal,
92.17% for stone and 98.55% for wake-up, versus only 55.81% for iron-pickaxe
crafting and 36.23% for iron-sword crafting. Seed 50/51 rarity estimates ranked
iron tools and wake-up low, but misplaced skeleton defeat ahead of iron
collection; seed 52 split iron-pickaxe events across two groups and lost an
iron-sword-majority group. This is weaker than the previous fixed-10B-probe
test and does not justify reliable all-22 online rare-event selection yet.

`grouping_summary.csv` and `summary.json` summarize results;
`evaluation_achievement_counts.csv` gives all 22 true event counts (audit only).
Each `groups/{representation}_seed{seed}/` contains a label-free `model.npz`,
numeric `group_membership.npz`, evaluation-only per-achievement precision/recall
and group-purity/frequency tables. Frequency estimates use deployable
single-event prototype assignments over normal reference episodes, not just
accepted POT matches. Ground-truth naming is never stored in inference models.
Raw-observation inference reproduced all 17,377 cached prototype assignments,
including every held-out event; manual actor/critic extraction matched the
network outputs exactly. The event bank itself is approximately 2.9 MiB
compressed; the optional actor/critic feature cache is approximately 59 MiB.

```bash
CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python \
  concept_mapping/group_own_policy_achievement_events.py \
  --out concept_mapping/runs/transport_own_policy_50m_seed50_1000episodes
```

Use `--stage group` to refit on the existing bank without new episodes, or a
new output directory to collect a new bank. `--run`/`--checkpoint`, collection
seed, split seed and grouping seeds are configurable. To classify new event
pairs without labels, use `compare_transport_achievement_groups.py` with
`--predict_model`, `--predict_input` (NPZ with before/after/rewards) and a new
`--predict_output` CSV. A static observation alone is not an event input.

## Cluster-free practice: from-scratch 10M pilot, two seeds per condition

Outputs: `runs/cluster_free_practice_10m/`. Training seeds 50 and 51, each with
fresh baseline PPO, uniform-start practice and prediction-error-prioritized
practice (six runs, GPUs 0–5). No restored 50M weights, learned event groups,
achievement-ID allocation, modified maps/rewards or auxiliary losses.
`experiment_plan.json` records settings; each run writes its command,
training/evaluation logs, progress and eventual manifest. Runs log to W&B in
`Craftax_Baselines`, names `cluster-free-{condition}-10m-seed{seed}`.

All three conditions use the same standard-reset wrapper and ordinary PPO:
128 environments, 64 steps, four epochs, eight minibatches, gamma .99,
GAE lambda .8, clipping .2, value coefficient .5, entropy .01, gradient norm
1, learning rate .0002 linearly annealed over the 10M budget, 512-unit tanh
actor/critic, one repeat. Normal training only through 1,007,616 interactions
(first completed PPO update reaching 1M). Initialization/RNG and warm-up are
matched within training seed; saved warm-up policy weights can be checked
using `compare_cluster_free_pilot.py --verify-warmup`.

Both practice arms reserve 13/128 lanes after warm-up (maximum 10.15625% of
interactions thereafter, approximately 9.13% of the whole budget; actual
fraction is recorded). Empty/exhausted banks fall back to normal resets.
Baseline uses no practice lanes but collects the same diagnostic bank.
Continuous banking uses NORMAL episodes only. Detect new achievement-reward
transitions by `reward - .1*health_change > .5`; do not read achievement IDs.
Within each normal episode, reservoir-select one eligible simulator state
exactly five actions before a detected reward, and admit it on episode end.
This episode-diverse bank avoids admitting every common reward from each
episode. It has 2,048 FIFO slots. Actual simulator flags/inventory/map are
retained without edits, but flags do not select/prioritize entries. Banks
diverge naturally once policies diverge; only their banking rules are matched.

Uniform practice samples usable entries equally. Prioritized practice uses
80% positive rank weights on smoothed absolute return-prediction error plus
20% uniform sampling. Initial scores equal 1, random ties break bank-position
bias, the first observed error replaces initialization, and subsequent scores
use EMA alpha .2. Each snapshot generation can start at most 30 drills;
sequential reset selection enforces the cap even for simultaneous resets.
Replacement generations prevent a completed old drill updating a new entry.
Practice lasts at most 30 actions, NOT until an achievement occurs. No
achievement-specific success label or artificial success reward is used.

For each fresh drill, compare the factual starting prediction V(s) with
G_hat = sum_{k=0}^{L-1} gamma^k r_k + gamma^L V(s_L), including the bootstrap
only for an artificial timeout, not true environment termination. Score
abs(G_hat - V(s)). True pre-reset observations supply the bootstrap. PPO
adds gamma*V(s_next) to the timeout transition's training target via reward
augmentation while done cuts GAE across the reset; original environment
rewards and logged normal returns remain unchanged. Priorities change
sampling, not PPO minibatch weights or reward. Errors are only a heuristic
for learning potential: noise or correctly predicted failures can make them
misleading, so this pilot does not assume rarity or concept discovery.

Requested budget is 10M per run, actual 9,994,240 because 8,192-interaction
updates are completed in full. Save policies at 65,536, 983,040, 1,007,616,
2,007,040, 5,005,312, 7,503,872 and final 9,994,240. Each worker automatically
evaluates frozen 5M and final policies on 512 NEW NORMAL episodes, seed 6050,
10,000-step default cap, stochastic actions. Reset-world keys depend only on
episode UID and seed, not when a policy terminates; dynamics/action streams
are also keyed per episode and timestep. Thus the same initial worlds are
used across policies and checkpoints, without drill starts or training banks.

Primary endpoint: mean normal-episode return. Report all 22 achievement
rates, episode lengths, actual practice fractions, snapshot diversity/reuse,
and errors. Keep per-training-seed results as well as the two-seed mean.
`compare_cluster_free_pilot.py` writes comparison CSV/JSON and paired normal
episode-bootstrap intervals. These intervals are conditional on trained
policy pairs; two seeds are a screening experiment, not robust across-seed
uncertainty. Prioritized versus uniform isolates allocation; prioritized
versus baseline tests the full method's transfer/sample efficiency.

Scripts: `cluster_free_practice.py`, `run_cluster_free_pilot.py`,
`evaluate_cluster_free_pilot.py`, `compare_cluster_free_pilot.py` in repo root.
Both practice variants are opt-in via PPO's `--cluster_free_practice`;
without that flag existing experiments/environment behavior are unchanged.
Eleven wrapper/regression tests passed. A separate 65,536-interaction offline
integration smoke run (seed9494, NOT one of the study runs) completed 425
drills and verified finite prediction-error scores and archive saving; the
new normal-world evaluator also completed a 32-episode smoke evaluation.
An initial slow-bookkeeping attempt was stopped during normal-only warm-up;
its output folders are preserved under `initial_slow_attempts/` and its W&B
runs/checkpoints remain available. Those partial attempts are excluded from
the study comparisons. The six complete runs restart from random seed after
vectorizing FIFO writes and limiting sequential reuse/score updates to
practice lanes; snapshot/sampling semantics and experimental settings are
unchanged. The same eleven tests passed again after optimization.

```bash
# One worker (launch in a persistent session; refuse existing output dirs):
/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python \
  run_cluster_free_pilot.py --condition prioritized --seed 50 --gpu 2

# Read current results / verify matched warm-up after those policies exist:
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python \
  compare_cluster_free_pilot.py --verify-warmup
```

## Symbolic recurrent PPO with a −2 death cost, 10B steps

Run records: `runs/classic_rnn_death_penalty2_10b_seed50/` (training log,
progress, metadata, and final manifest). Reusable launcher:
`run_classic_rnn_death_penalty2_10b.py --gpu 0`; exact command:
`run_classic_rnn_death_penalty2_10b.sh` (both in repo root).

Fresh seed50 training uses `ppo_rnn_og.py`, the symbolic-observation recurrent
trainer, NOT the CNN/pixel trainer `ppo_rnn.py`. A 512-unit GRU feeds the actor
and critic. Match the feedforward death-cost run's core configuration: 1,024
environments, 64 rollout steps, four epochs, eight recurrent minibatches,
gamma .99, GAE lambda .8, clipping .2, value coefficient .5, entropy .01,
learning rate .0002 linearly annealed over 10B interactions, gradient norm 1.
Optimistic resets remain enabled with ratio16, as in that feedforward run.
This is one experiment, not a new matched RNN baseline comparison.

Reuse the opt-in `ClassicDeathPenaltyWrapper` before logging/reset wrappers.
Subtract an ADDITIONAL 2 on true death, including lava entry, while retaining
the environment's original health/achievement rewards. Time-limit-only
endings receive no additional cost. Installed Craftax code and default
baseline rewards are unchanged. The flag defaults to zero in the RNN trainer.

W&B project `Craftax_Baselines`, run name
`classic-rnn-death-penalty2-10b-seed50`. Retain the same logarithmic requested
checkpoint milestones as the feedforward −2 run, from 65,536 to 7.5B, rounded
up to completed 65,536-interaction updates. Save the actual final checkpoint
at 9,999,941,632 interactions (10B budget rounded down to whole updates).
Checkpoint arithmetic uses Python integers to avoid 32-bit overflow.

Validation: two focused tests cover additive death/lava cost, healthy
time-limit exemption and recurrent state resets. A separate offline 131,072-
interaction smoke run used the FULL 1,024-env/512-GRU shape and successfully
saved intermediate and final TrainState checkpoints.

### Milder recurrent death-cost ablation: −0.15

Fresh seed50, 10B-step symbolic GRU run with every training setting and
checkpoint milestone unchanged from the −2 RNN experiment. Change ONLY the
additional death cost to −0.15 (including lava); ordinary health/achievement
rewards and time-limit-only endings are unchanged. No installed environment
files are modified. The original −2 run is not stopped or overwritten.
Launch on GPU 1 with:

```bash
/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python \
  run_classic_rnn_death_penalty2_10b.py --gpu 1 --death_penalty 0.15
```

The launcher now accepts the penalty magnitude; its default remains 2.0.
Records: `runs/classic_rnn_death_penalty0p15_10b_seed50/`. W&B run name:
`classic-rnn-death-penalty0p15-10b-seed50`. Metadata records the exact launch
arguments and shell environment overrides. Tests cover both magnitudes,
including the fact that the death cost is additive and not charged every
step or on a healthy time limit.

### Recurrent death cost −0.5, with and without diamond reward +5

Two fresh 10B-budget seed50 symbolic-GRU runs, all PPO settings, optimistic
reset settings and logarithmic milestones unchanged:

| Run directory under `runs/` | Additional death cost | First diamond achievement | GPU |
| --- | ---: | ---: | ---: |
| `classic_rnn_death_penalty0p5_10b_seed50/` | −0.5 | +1 (original) | 0 |
| `classic_rnn_death_penalty0p5_diamond5_10b_seed50/` | −0.5 | +5 total | 2 |

The diamond variant uses an opt-in `ClassicDiamondRewardWrapper` before
logging and reset wrappers. Add exactly +4 on the false-to-true transition
of `COLLECT_DIAMOND`, preserving the original +1 and any health/death reward.
Subsequent diamonds get no extra reward, as the base environment awards
achievements only once per episode. All other rewards, observations, game
mechanics and time-limit handling remain unchanged; no installed Craftax
files are edited. Both runs penalize death; neither gives a positive reward
for dying. Existing −0.15/−2 runs are not overwritten or interrupted.

```bash
/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python \
  run_classic_rnn_death_penalty2_10b.py --gpu 0 --death_penalty 0.5
/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python \
  run_classic_rnn_death_penalty2_10b.py --gpu 2 --death_penalty 0.5 --diamond_reward 5
```

Default `--diamond_reward 1` preserves previous behavior. Metadata and W&B
record both reward settings; compare diamond success rates rather than
interpreting higher shaped return alone as better behavior across reward
scales. The RNN episode renderer applies both configured reward wrappers.
Five focused tests passed, including +5 versus +6, no repeat bonus, unchanged
other rewards, combined death/health rewards, and default equivalence. A real
Craftax transition check confirmed first-diamond reward 1→5 with identical
successor observations and no extra reward on the second diamond.

## ~20M feedforward PPO, original rewards versus −2 death cost

Launcher: `run_large_ppo_20m_10b.py`; plans in
`runs/large_feedforward_20m_10b_seed50/{baseline,death_penalty2}/`.
Both runs launched after storage was freed (about 850 GB available).
Baseline uses GPU 3, tmux session `ff20m_baseline_10b`, and W&B run
`synaderi-uc-davis/Craftax_Baselines/bxl5lhe7`. The −2 death-cost condition
uses GPU 4, tmux session `ff20m_death2_10b`, and W&B run
`synaderi-uc-davis/Craftax_Baselines/fmv1otdn`.
The launcher refuses to start below 30 GiB free and never prunes existing experiments.

Use the existing feedforward actor/critic with three tanh hidden layers per
branch, increasing width from 512 to 1,920. Exact combined parameter count
19,956,498: actor 9,993,617, critic 9,962,881. No GRU or shared encoder.
Fresh seed50 training, 10B requested interactions (9,999,941,632 actual),
1,024 environments, 64 steps, four epochs, eight minibatches; all core PPO
settings and logarithmic milestones match the small feedforward runs.
Optimistic resets remain enabled at ratio16. Baseline rewards are untouched;
the second condition adds only a −2 death cost, including lava but not
time-limit-only termination. Diamond remains +1 in both conditions.

Parameter counting was validated against the model's abstract initialized
tree. A non-saving, no-W&B 131,072-interaction GPU test passed with the FULL
1,024-env/64-step/width1,920 configuration. Command comparison confirmed that
only death cost and run name differ. Plans capture source hashes, commands,
parameter counts, checkpoint labels and storage requirements.

After providing sufficient persistent storage, launch each worker in a
persistent session on an available GPU:

```bash
/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python \
  run_large_ppo_20m_10b.py --condition baseline --gpu 3
/home/shawheen/miniconda3/envs/jax-v100-cuda12/bin/python \
  run_large_ppo_20m_10b.py --condition death_penalty2 --gpu 4
```

W&B, full policy/optimizer checkpoints, local logs and final manifests are
enabled when training launches. `--prepare-only` writes plans without
training; it does not bypass the storage guard for actual runs.

## MLP, GRU, and Transformer PPO: 10M-step pilot

See [the experiment setup](runs/ppo_memory_comparison_10m/README.md) for nine
fresh standard-reward runs (three architectures × seeds 50/51/52), GPU queues,
checkpoint schedules, and matched frozen-policy evaluations at 1M/5M/10M.
The Transformer has two layers, 256 embedding dimensions, four attention
heads, and an exact 64-observation window. Launcher:
`run_ppo_memory_comparison.py`; reports are generated automatically by
`summarize_ppo_memory_comparison.py` as evaluations finish.

## Symbolic EfficientZero-style pilot: 10M real interactions

See [the setup and validation](runs/symbolic_ez_10m/README.md) for three fresh
standard-reward seeds50/51/52, learned symbolic dynamics with Gumbel search,
matched PPO controls, frozen evaluations, and model-quality diagnostics.
Selected EfficientZeroV2 components were translated to JAX; upstream source,
licenses, and differences are recorded in repository-root `SYMBOLIC_EZ_NOTICE.md`.
Six behavioral tests plus small/full-size real training and checkpoint/evaluation
smokes passed before launch. Launchers are `train_symbolic_ez.py` and
`run_symbolic_ez_pilot.py`; reports use `summarize_symbolic_ez_pilot.py`.

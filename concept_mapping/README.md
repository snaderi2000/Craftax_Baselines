# Concept Mapping

Offline concept discovery for symbolic Craftax transitions.

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

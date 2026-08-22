# Concept Mapping

Offline concept discovery for symbolic Craftax transitions.

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


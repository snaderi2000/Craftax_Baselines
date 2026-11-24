"""Experience memory and retrieval for Cumulative Culture PPO.

The retrieval encoder is fixed across memory creation and querying. It is
separate from the current critic and never supplies a value target.
"""
from dataclasses import dataclass
import json
import numpy as np
import jax
import jax.numpy as jnp


@dataclass
class CultureEntry:
    start_observation: np.ndarray
    action_sequence: np.ndarray
    rewards: np.ndarray
    discounted_return: float
    generation_id: int


def encode(params, observations, mode="critic_hidden"):
    if mode == "raw_flattened_observation":
        z = observations
    else:
        root = params["params"]
        offset = 4 if mode == "critic_hidden" else 0
        z = observations
        for i in range(offset, offset + 3):
            z = jnp.tanh(z @ root[f"Dense_{i}"]["kernel"] + root[f"Dense_{i}"]["bias"])
    return z / jnp.maximum(jnp.linalg.norm(z, axis=-1, keepdims=True), 1e-8)


class CultureSelector:
    @staticmethod
    def score_fragment(returns, embeddings, return_weight=1., novelty_weight=.25):
        lo, hi = np.percentile(returns, [5, 95])
        outcomes = np.clip((returns-lo)/max(hi-lo, 1e-8), 0, 1)
        reference = embeddings[np.linspace(0, len(embeddings)-1, min(256,len(embeddings)), dtype=int)]
        # Mean neighbor distance is stable even when the reference includes self.
        similarity = embeddings @ reference.T
        k = min(4, len(reference))
        novelty = 1 - np.partition(similarity, -k, axis=1)[:, -k:].mean(axis=1)
        return return_weight*outcomes + novelty_weight*novelty, outcomes, novelty


class CultureMemory:
    def __init__(self, arrays, metadata):
        self.arrays, self.metadata = arrays, metadata

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            arrays = {k: d[k] for k in d.files if k != "metadata"}
            metadata = json.loads(str(d["metadata"]))
        return cls(arrays, metadata)

    def save(self, path):
        np.savez_compressed(path, **self.arrays, metadata=json.dumps(self.metadata))

    def add_entries(self, candidate_arrays):
        self.arrays = {k: np.concatenate([v, candidate_arrays[k]], axis=0)
                       for k,v in self.arrays.items()}

    def consolidate(self, max_size=512, duplicate_similarity_threshold=.995,
                    entries_per_generation=512):
        a=self.arrays
        # Prefer a better return whenever two starts are near duplicates.
        order=np.lexsort((-a["returns"], -a["scores"]))
        keep=[]
        for idx in order:
            if keep:
                sims=a["embeddings"][keep] @ a["embeddings"][idx]
                matches=np.flatnonzero(sims > duplicate_similarity_threshold)
                if len(matches):
                    match=int(matches[np.argmax(sims[matches])])
                    if a["returns"][idx] > a["returns"][keep[match]]:
                        keep[match]=int(idx)
                    continue
            keep.append(int(idx))
            if len(keep)>=min(max_size,entries_per_generation):
                break
        self.arrays={k:v[keep] for k,v in a.items()}

    def retrieve(self, embeddings, k=4):
        similarity=embeddings @ jnp.asarray(self.arrays["embeddings"]).T
        return jax.lax.top_k(similarity, min(k,len(self.arrays["embeddings"])))


def build_policy_target(memory, similarities, indices, action_dim=17,
                        threshold=.95, temperature=.1, weighting="similarity_return",
                        shuffled_actions=None):
    valid=similarities >= threshold
    success=jnp.asarray(memory.arrays["outcomes"])[indices]
    actions=jnp.asarray(memory.arrays["actions"][:,0] if shuffled_actions is None
                        else shuffled_actions)[indices]
    sim_weights=jnp.exp((similarities-jnp.max(similarities,axis=-1,keepdims=True))/temperature)
    if weighting == "similarity":
        weights=sim_weights*valid
    elif weighting == "return":
        weights=success*valid
    else:
        weights=sim_weights*success*valid
    mass=weights.sum(axis=-1)
    q=(jax.nn.one_hot(actions,action_dim)*weights[...,None]).sum(axis=1)/jnp.maximum(mass[:,None],1e-8)
    confidence=jnp.where(mass>0,jnp.clip(similarities[:,0],0,1),0)
    historical_success=(success*weights).sum(axis=-1)/jnp.maximum(mass,1e-8)
    gate=confidence*historical_success
    mean_return=(jnp.asarray(memory.arrays["returns"])[indices]*weights).sum(axis=-1)/jnp.maximum(mass,1e-8)
    return q, gate, dict(match_rate=(mass>0).mean(), similarity=similarities[:,0].mean(),
                        confidence=confidence.mean(), historical_return=mean_return.mean())

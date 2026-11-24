"""One action distribution for PPO collection, likelihoods, and entropy."""
import math
import distrax


def tempered_policy(distribution,temperature=1.):
    if not math.isfinite(temperature) or temperature<=0:
        raise ValueError('Training temperature must be finite and strictly positive')
    if temperature==1.:return distribution
    return distrax.Categorical(logits=distribution.logits_parameter()/temperature)

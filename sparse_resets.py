"""Generate fresh worlds only for terminal environments, with their original RNG keys."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

from wrappers import AutoResetEnvWrapper, BatchEnvWrapper


class SparseResetBatch(BatchEnvWrapper):
    def __init__(self, env, num_envs):
        if not isinstance(env, AutoResetEnvWrapper):
            raise ValueError("The comparator requires the ordinary per-world autoreset")
        super().__init__(env._env, num_envs)
        self.mesh = Mesh(np.array(jax.local_devices()), ('ppo_data',))

    def step_local(self, keys, state, action, params):
        split = jax.vmap(jax.random.split)(keys)
        obs, state, reward, done, info = self.step_fn(split[:, 1], state, action, params)
        count = keys.shape[0]
        chunk = min(8, count)
        indices = jnp.nonzero(done, size=count, fill_value=count)[0]
        num_done = done.sum(dtype=jnp.int32)

        def reset_chunk(i, carry):
            obs, state = carry
            positions = i * chunk + jnp.arange(chunk)
            ids = jnp.where(positions < num_done, indices[positions], count)
            # The baseline splits the unused first key once more for reset.
            reset_keys = jax.vmap(jax.random.split)(split[ids.clip(0, count-1), 0])[:, 1]
            new_obs, new_state = self.reset_fn(reset_keys, params)
            def scatter(old, new):
                return old.at[ids].set(new, mode='drop')
            return scatter(obs, new_obs), jax.tree_util.tree_map(scatter, state, new_state)

        obs, state = jax.lax.fori_loop(0, (num_done + chunk - 1) // chunk,
                                       reset_chunk, (obs, state))
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        _, key = jax.random.split(rng)
        keys = jax.random.split(key, self.num_envs)
        # Keep simulator parameters static in the local single-device function.
        local_step = partial(self.step_local, params=params)
        run = shard_map(local_step, mesh=self.mesh, in_specs=(P('ppo_data'),) * 3,
                        out_specs=P('ppo_data'), check_rep=False)
        return run(keys, state, action)

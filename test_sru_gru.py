"""Check the SRU transfer and freeze boundary independently of rollout noise."""

import unittest

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np

from models.sru_gru import (
    apply_scaled_updates, gate_penalty, learning_rate_scale, make_optimizer,
    mask_warmup_gradients, transplant_gru,
)
from ppo_rnn_og import ActorCriticRNN, initialize_recurrent_carry


class SruGruTest(unittest.TestCase):
    def test_normalized_residual_heads_share_the_observation_and_memory(self):
        cfg = dict(LAYER_SIZE=16, SRU_GRU=True, GRU_CONCAT_OBSERVATION=True, SRU_RESIDUAL_ACTOR_CRITIC=True)
        net = ActorCriticRNN(17, cfg)
        key = jax.random.PRNGKey(52)
        obs = jax.random.normal(key, (3, 4, 19))
        resets = jnp.ones((3, 4), bool)
        hidden = initialize_recurrent_carry(4, cfg)
        params = net.init(key, hidden, (obs, resets))
        h, pi, value = net.apply(params, hidden, (obs, resets))
        h_alt, pi_alt, v_alt = net.apply(params, hidden + 1, (obs, resets))
        self.assertEqual(pi.logits.shape, (3, 4, 17))
        self.assertEqual(value.shape, (3, 4))
        for a, b in ((h,h_alt), (pi.logits,pi_alt.logits), (value,v_alt)):
            np.testing.assert_array_equal(a,b)
        for name in ('actor_head','critic_head'):
            head = params['params'][name]
            self.assertEqual(head['Dense_0']['kernel'].shape, (32, 16))
            self.assertIn('DenseResBlock_0', head)
            self.assertIn('DenseResBlock_1', head)

    def test_identity_transfer_and_learnable_candidate(self):
        config = {'LAYER_SIZE': 16, 'GRU_CONCAT_OBSERVATION': True}
        key = jax.random.PRNGKey(19)
        hidden = initialize_recurrent_carry(2, config)
        obs = jax.random.normal(key, (6, 2, 19))
        dones = jnp.zeros((6, 2), dtype=bool).at[3, 0].set(True)
        baseline = ActorCriticRNN(17, config)
        augmented = ActorCriticRNN(17, dict(config, SRU_GRU=True))
        original = baseline.init(key, hidden, (obs, dones))
        params = transplant_gru(augmented.init(key, hidden, (obs, dones)), original)
        old_h, old_pi, old_v = baseline.apply(original, hidden, (obs, dones))
        new_h, new_pi, new_v = augmented.apply(params, hidden, (obs, dones))
        for old, new in [(old_h, new_h), (old_pi.probs, new_pi.probs), (old_v, new_v)]:
            np.testing.assert_allclose(new, old, rtol=1e-6, atol=1e-6)
        self.assertEqual(float(gate_penalty(params)), 0.)
        gradient = jax.grad(lambda p: augmented.apply(p, hidden, (obs, dones))[2].sum())(params)
        self.assertGreater(float(jnp.linalg.norm(gradient['params']['ScannedRNN_0']['spatial_transform']['kernel'])), 0.)

    def test_pretrained_optimizer_waits_until_after_warmup(self):
        config = dict(NUM_ENVS=4, NUM_STEPS=4, NUM_MINIBATCHES=2, UPDATE_EPOCHS=1,
                      SRU_WARMUP_STEPS=17, SRU_GATE_LR=.005, MAX_GRAD_NORM=1.)
        params = {'params': {'leaf': jnp.asarray(1.), 'ScannedRNN_0': {'spatial_transform': {'bias': jnp.ones(2)}}}}
        state = TrainState.create(apply_fn=None, params=params, tx=make_optimizer(config, .001))
        partition = lambda st: st.opt_state[1].inner_states['pretrained'].inner_state
        moments_before = partition(state).inner_state
        gradients = jax.tree_util.tree_map(jnp.ones_like, params)

        @jax.jit
        def update(st):
            return st.apply_gradients(grads=mask_warmup_gradients(gradients, st.step, config))

        for _ in range(4):
            state = update(state)
            np.testing.assert_array_equal(state.params['params']['leaf'], 1.)
        for before, during in zip(jax.tree_util.tree_leaves(moments_before),
                                  jax.tree_util.tree_leaves(partition(state).inner_state)):
            np.testing.assert_array_equal(before, during)
        self.assertLess(float(state.params['params']['ScannedRNN_0']['spatial_transform']['bias'][0]), 1.)
        state = update(state)
        self.assertLess(float(state.params['params']['leaf']), 1.)

    def test_annealing_scales_both_partitions_without_resetting_adam(self):
        start = 10_000_000_000
        config = dict(NUM_ENVS=4, NUM_STEPS=4, NUM_MINIBATCHES=2, UPDATE_EPOCHS=1,
                      SRU_WARMUP_STEPS=0, SRU_GATE_LR=.01, MAX_GRAD_NORM=1.,
                      START_TIMESTEPS=start, SRU_LR_ANNEAL_FROM=start,
                      SRU_LR_ANNEAL_UNTIL=start + 160, SRU_LR_FINAL_FRACTION=.1)
        schedule = jax.jit(lambda update: learning_rate_scale(config, update))
        np.testing.assert_allclose([schedule(i) for i in (0, 5, 10, 20)], [1., .55, .1, .1])
        resumed = dict(config, START_TIMESTEPS=start + 80)
        np.testing.assert_allclose(learning_rate_scale(resumed, 0), schedule(5))
        params = {'params': {'leaf': jnp.asarray(1.), 'ScannedRNN_0': {'spatial_transform': {'bias': jnp.ones(2)}}}}
        grads = jax.tree_util.tree_map(jnp.ones_like, params)
        state = TrainState.create(apply_fn=None, params=params, tx=make_optimizer(config, .01))
        state = state.apply_gradients(grads=grads)
        reference = jax.jit(lambda st: st.apply_gradients(grads=grads))(state)
        scaled = jax.jit(apply_scaled_updates)(state, grads, schedule(5))
        for full, scaled_leaf, original in zip(jax.tree_util.tree_leaves(reference.params),
                                               jax.tree_util.tree_leaves(scaled.params),
                                               jax.tree_util.tree_leaves(state.params)):
            np.testing.assert_allclose(scaled_leaf-original, .55*(full-original), atol=1e-7)
        for full, actual in zip(jax.tree_util.tree_leaves(reference.opt_state),
                                jax.tree_util.tree_leaves(scaled.opt_state)):
            np.testing.assert_allclose(actual, full, rtol=2e-7, atol=0.)
        self.assertEqual(int(scaled.step), int(reference.step))


if __name__ == '__main__':
    unittest.main()

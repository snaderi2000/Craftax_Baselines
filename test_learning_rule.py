from copy import deepcopy
import random
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from learning_rule.graphs import Builder, INPUTS, PPO, Program, graph_types
from learning_rule.evolution import mutate, numerical_screen
from learning_rule.runtime import credit_targets, execute, program_loss, update_memory


class LearningRuleTests(unittest.TestCase):
    def test_ppo_loss_and_every_gradient_match_independent_reference(self):
        x = jnp.arange(32, dtype=jnp.float32)
        old_logp = -.2 * x - 1.
        old_value = jnp.cos(x)
        adv, target = jnp.sin(x), jnp.cos(x * .3)
        def reference(logp, value, entropy):
            clipped_value = old_value + (value - old_value).clip(-.2, .2)
            critic = .5 * jnp.maximum(jnp.square(value - target), jnp.square(clipped_value - target)).mean()
            ratio = jnp.exp(logp - old_logp)
            normalized = (adv - adv.mean()) / (adv.std() + 1e-8)
            actor = -jnp.minimum(ratio * normalized, jnp.clip(ratio, .8, 1.2) * normalized).mean()
            return actor + .5 * critic - .01 * entropy.mean()
        def graph_loss(logp, value, entropy):
            return program_loss(PPO.to_dict(), logp, old_logp, value, old_value,
                adv, target, entropy, jnp.zeros(4), .99, .8, .1, .2, .5, .01)
        args = (old_logp + .4 * jnp.sin(x), old_value + .5 * jnp.sin(x), jnp.ones(32) * 2.7)
        first = jax.value_and_grad(reference, argnums=(0, 1, 2))(*args)
        second = jax.jit(jax.value_and_grad(graph_loss, argnums=(0, 1, 2)))(*args)
        for a, b in zip(jax.tree_util.tree_leaves(first), jax.tree_util.tree_leaves(second)):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)

    def test_credit_matches_gae_and_masks_terminal_bootstrap(self):
        rewards = np.arange(18, dtype=np.float32).reshape(6, 3) / 10
        values = np.cos(rewards).astype(np.float32)
        done = np.zeros((6, 3), bool); done[2, 0] = True; done[-1, 1] = True
        last = np.array([10., 20., 30.], np.float32)
        expected = np.zeros_like(values); carry = np.zeros(3, np.float32); nxt = last
        for t in reversed(range(6)):
            delta = rewards[t] + .99 * nxt * (1 - done[t]) - values[t]
            carry = delta + .99 * .8 * (1 - done[t]) * carry
            expected[t] = carry; nxt = values[t]
        actual, target = credit_targets(PPO.to_dict(), jnp.array(rewards), jnp.array(values),
            jnp.array(done), jnp.array(last), jnp.zeros(4), .99, .8, 0.)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-6)
        np.testing.assert_allclose(target, expected + values, rtol=2e-6, atol=1e-6)
        changed = rewards.copy(); changed[3:, 0] += 100
        isolated, _ = credit_targets(PPO.to_dict(), jnp.array(changed), jnp.array(values),
            jnp.array(done), jnp.array(last), jnp.zeros(4), .99, .8, 0.)
        np.testing.assert_array_equal(actual[:3, 0], isolated[:3, 0])

    def test_stop_gradient_is_a_real_mutable_operation(self):
        graph = dict(nodes=[dict(id='y', op='detach', args=['x'])], outputs=['y'])
        self.assertEqual(float(execute(graph, dict(x=jnp.array(2.)))[0]), 2.)
        self.assertEqual(float(jax.grad(lambda x: execute(graph, dict(x=x))[0])(jnp.array(2.))), 0.)

    def test_memory_program_updates_state_and_does_not_backpropagate(self):
        p = PPO.to_dict(); b = Builder(); half = b.const(.5)
        mean = b.op('mul', half, b.op('add', 'm0', 'reward_mean'))
        p['memory'] = b.finish(mean, 'm1', 'm2', 'm3')
        p = Program.from_dict(p).to_dict()
        def update(memory):
            return update_memory(p, memory, jnp.ones((6, 2)), jnp.zeros((6, 2)),
                jnp.zeros((6, 2), bool), jnp.zeros((6, 2)), .99, .8, .5)
        np.testing.assert_allclose(update(jnp.zeros(4)), [.5, 0, 0, 0])
        np.testing.assert_array_equal(jax.jacrev(update)(jnp.zeros(4)), np.zeros((4, 4)))

    def test_validation_rejects_cycles_unknown_operators_vector_loss_and_extra_code(self):
        for modification in ('cycle', 'operator', 'loss', 'code'):
            p = PPO.to_dict()
            if modification == 'cycle': p['loss']['nodes'][0] = dict(id='x', op='add', args=['x', 'value'])
            elif modification == 'operator': p['loss']['nodes'][0]['op'] = 'eval'
            elif modification == 'loss': p['loss']['outputs'] = ['value']
            else: p['python'] = 'pass'
            with self.assertRaises(ValueError): Program.from_dict(p)

    def test_mutations_edit_graph_structure_with_provenance_and_no_parent_changes(self):
        original = PPO.encoded; rng = random.Random(99); kinds = set(); stages = set()
        for _ in range(100):
            child, change = mutate(PPO, rng)
            kinds.add(change['operator']); stages.add(change['stage'])
            self.assertNotEqual(child.id, PPO.id)
            self.assertEqual(Program.from_dict(child.to_dict()), child)
            self.assertEqual(change['parent'], PPO.id)
            self.assertEqual(change['child'], child.id)
            for stage in INPUTS: graph_types(child.to_dict()[stage], INPUTS[stage])
        self.assertEqual(PPO.encoded, original)
        self.assertEqual(kinds, {'insert', 'rewire', 'replace', 'delete'})
        self.assertEqual(stages, set(INPUTS))
        self.assertTrue(numerical_screen(PPO)['valid'])

    def test_dead_nodes_do_not_create_new_genotypes(self):
        p = PPO.to_dict()
        p['loss']['nodes'].append(dict(id='unused', op='constant', value=10))
        self.assertEqual(Program.from_dict(p), PPO)


if __name__ == '__main__':
    unittest.main()

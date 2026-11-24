import random
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from aux_prediction import TASKS, initialize_head, loss, projection, sample_batch, targets
from aux_search.recipes import BASELINE, Recipe, initial_population, mutate
from models.actor_critic import ActorCritic
from run_auxiliary_search import initialize
from run_auxiliary_search import Search, TimeLimitReached, stop_worker_group, wait_for_free_gpu


class RecipeTests(unittest.TestCase):
    def test_initial_population_reproducible_and_covers_tasks(self):
        population = initial_population(2026)
        self.assertEqual(population, initial_population(2026))
        self.assertEqual(len({r.id for r in population}), 8)
        self.assertEqual({r.task for r in population}, set(TASKS))

    def test_every_mutation_changes_one_valid_field_and_records_it(self):
        rng = random.Random(2026)
        for parent in initial_population(2026):
            for _ in range(100):
                child, change = mutate(parent, rng)
                changed = [f for f in ('task', 'horizon', 'weight') if getattr(child, f) != getattr(parent, f)]
                self.assertEqual(changed, [change['field']])
                self.assertNotEqual(child.id, parent.id)
                if child.task == 'inverse':
                    self.assertEqual(child.horizon, 1)
        with self.assertRaises(ValueError):
            mutate(BASELINE, rng)

    def test_weight_boundaries_and_horizon_neighbors(self):
        rng = random.Random(0)
        for weight in (.001, 1.):
            child, _ = mutate(Recipe('event', 4, weight), rng, 'weight')
            self.assertEqual(child.weight, .002 if weight == .001 else .5)
        child, _ = mutate(Recipe('event', 1, .1), rng, 'horizon')
        self.assertEqual(child.horizon, 4)

    def test_prepare_resume_preserves_population_and_zero_weight_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = initialize(tmp)
            self.assertEqual(first, initialize(tmp))
            self.assertFalse(first['weight_inheritance'])
            self.assertLess(first['max_total_interactions'], 200000000)
            with self.assertRaises(ValueError):
                initialize(tmp, 1)

    def test_controller_generates_full_search_and_matched_random_branch(self):
        import math
        with tempfile.TemporaryDirectory() as tmp:
            search = Search(tmp, 0, 2026)
            seen = {}
            def fake_evaluation(recipe, seed, confirmation=False):
                key = recipe.id, seed
                if key not in seen:
                    seen[key] = 0. if recipe.task == 'none' else 3. - abs(math.log10(recipe.weight) + 1)
                return seen[key]
            search.evaluate_recipe = fake_evaluation
            search.prefetch = lambda recipes, confirmation=False: None
            search.run()
            self.assertEqual(search.data['state'], 'complete')
            self.assertEqual(len(search.candidates('mutation')), 32)
            self.assertEqual(len(search.candidates('random')), 32)
            self.assertEqual(len(seen), 166)
            self.assertEqual(len(search.data['confirmation']), 4)
            for child in search.data['candidates']:
                if child['controller'] == 'mutation':
                    parent = next(c for c in search.data['candidates'] if c['id'] == child['parent'])
                    changed = [k for k in child['recipe'] if child['recipe'][k] != parent['recipe'][k]]
                    self.assertEqual(changed, [child['mutation']['field']])

    def test_deadline_is_persisted_and_stops_before_starting_another_trial(self):
        import json
        import time
        with tempfile.TemporaryDirectory() as tmp:
            search = Search(tmp, 0, 2026, max_hours=10)
            deadline = search.data['run_deadline_at']
            self.assertAlmostEqual(search.remaining_seconds(), 36000, delta=2)
            resumed = Search(tmp, 0, 2026, max_hours=10)
            self.assertEqual(resumed.data['run_deadline_at'], deadline)
            search.data['run_deadline_at'] = time.time() - 1
            with self.assertRaises(TimeLimitReached):
                search.evaluate_recipe(BASELINE, 50)
            self.assertEqual(search.data['trials'], [])
            self.assertEqual(json.loads((search.out / 'progress.json').read_text())['state'], 'time_limit_reached')

    def test_worker_cleanup_targets_only_its_dedicated_process_group(self):
        import signal
        import subprocess
        from unittest.mock import Mock, patch
        process = Mock(pid=12345)
        process.wait.side_effect = [subprocess.TimeoutExpired('worker', 10), 0]
        with patch('run_auxiliary_search.os.killpg') as kill:
            stop_worker_group(process)
        self.assertEqual(kill.call_args_list[0].args, (12345, signal.SIGTERM))
        self.assertEqual(kill.call_args_list[1].args, (12345, signal.SIGKILL))

    def test_gpu_check_allows_previous_worker_utilization_to_settle(self):
        from unittest.mock import patch
        with patch('run_auxiliary_search.subprocess.check_output', side_effect=['16145, 50', '16145, 0']) as probe:
            with patch('run_auxiliary_search.time.sleep') as sleep:
                wait_for_free_gpu(0)
        self.assertEqual(probe.call_count, 2)
        sleep.assert_called_once_with(2)
        with patch('run_auxiliary_search.subprocess.check_output', return_value='500, 80'):
            with patch('run_auxiliary_search.time.sleep'):
                with self.assertRaisesRegex(RuntimeError, 'occupied'):
                    wait_for_free_gpu(0)

    def test_parallel_scheduler_reserves_one_worker_per_gpu_and_reuses_results(self):
        import json
        from pathlib import Path
        from unittest.mock import patch
        from aux_search.recipes import initial_population
        with tempfile.TemporaryDirectory() as tmp:
            search = Search(tmp, 0, 2026, gpus=[0, 1])
            recipes = initial_population(2026)[:2]
            occupied, launched, peak = set(), [], [0]

            class FakeProcess:
                def __init__(self, command, **kwargs):
                    self.gpu = int(kwargs['env']['CUDA_VISIBLE_DEVICES'])
                    self.pid = 50000 + len(launched)
                    self.polls = 0
                    if self.gpu in occupied:
                        raise AssertionError('GPU double-booked')
                    occupied.add(self.gpu); peak[0] = max(peak[0], len(occupied))
                    directory = Path(command[command.index('--out') + 1])
                    seed = int(command[command.index('--seed') + 1])
                    (directory / 'result.json').write_text(json.dumps(dict(complete=True,
                        training_steps=999424, environment_steps=2048, mean_return=float(seed))))
                    launched.append(command)

                def poll(self):
                    self.polls += 1
                    if self.polls == 1:
                        return None
                    occupied.discard(self.gpu)
                    return 0

            with patch.object(search, '_wait_gpu'):
                with patch('aux_search.parallel.subprocess.Popen', FakeProcess):
                    with patch('aux_search.parallel.time.sleep'):
                        search.prefetch(recipes)
                        self.assertEqual(search.fitness(recipes[0]), 50.5)
                        search.prefetch(recipes)
            self.assertEqual(len(launched), 4)
            self.assertEqual(peak[0], 2)
            self.assertEqual(len(search.data['trials']), 4)
            self.assertTrue(all(t['state'] == 'complete' for t in search.data['trials']))
            self.assertEqual(search.evaluation_charged(), 8192)
            self.assertEqual(search.interactions_charged(), 6008192)

    def test_parallel_scheduler_recovers_a_finished_handoff_worker_without_retraining(self):
        import json
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            search = Search(tmp, 0, 2026, gpus=[0, 1])
            key = f'{BASELINE.id}_seed50'
            search.data['trials'].append(dict(id=key, recipe_id=BASELINE.id, seed=50,
                state='running', gpu=0, pid=987654321, training_reserved=1000000,
                evaluation_reserved=250000, confirmation=False, started_at=0))
            directory = search.out / 'trials' / key
            directory.mkdir(parents=True)
            (directory / 'result.json').write_text(json.dumps(dict(complete=True,
                training_steps=999424, environment_steps=4096, mean_return=4.5)))
            with patch('aux_search.parallel.process_live', return_value=False):
                with patch('aux_search.parallel.subprocess.Popen') as spawn:
                    self.assertEqual(search.evaluate_recipe(BASELINE, 50), 4.5)
            spawn.assert_not_called()
            self.assertEqual(search.evaluation_charged(), 4096)


class PredictionTests(unittest.TestCase):
    def data(self):
        obs = jnp.arange(8 * 2 * 1345, dtype=jnp.float32).reshape(8, 2, 1345) / 10000
        next_obs = obs + .01
        actions = jnp.tile(jnp.arange(8)[:, None], (1, 2))
        rewards = jnp.zeros((8, 2)).at[1, 0].set(.1).at[2, 0].set(1.)
        rewards = rewards.at[2, 1].set(1.1).at[3, 1].set(1.)
        done = jnp.zeros((8, 2), bool).at[3, 0].set(True)
        return obs, next_obs, actions, rewards, done

    def test_windows_do_not_cross_resets_or_rollout_end(self):
        data = targets(*self.data(), 'event', 4, projection(1345, 2026))
        np.testing.assert_array_equal(data[-1][:, 0], [0, 0, 0, 0, 1, 0, 0, 0])
        np.testing.assert_array_equal(data[-1][:, 1], [1, 1, 1, 1, 1, 0, 0, 0])
        batch = sample_batch(data, jax.random.PRNGKey(1), 256)
        np.testing.assert_array_equal(batch[-1], np.ones(256))

    def test_event_is_exactly_plus_one_and_regression_is_after_minus_before(self):
        values = self.data()
        event = targets(*values, 'event', 1, projection(1345, 2026))
        self.assertEqual(float(event[3][1, 0, 0]), 0)
        self.assertEqual(float(event[3][2, 0, 0]), 1)
        self.assertEqual(float(event[3][2, 1, 0]), 0)
        regression = targets(*values, 'stats_delta', 4, projection(1345, 2026))
        np.testing.assert_allclose(regression[3][0, 1], values[1][3, 1, 1323:1339] - values[0][0, 1, 1323:1339])

    def test_heads_preserve_policy_and_value_and_train_only_actor_encoder(self):
        key = jax.random.PRNGKey(0)
        obs = self.data()[0][0]
        net = ActorCritic(17, 16)
        original = net.init(key, obs)
        p0, v0 = net.apply(original, obs)
        for task in TASKS:
            params = initialize_head(original, key, 16, task)
            pi, value = net.apply(params, obs)
            np.testing.assert_array_equal(pi.logits_parameter(), p0.logits_parameter())
            np.testing.assert_array_equal(value, v0)
            data = targets(*self.data(), task, 1, projection(1345, 2026))
            batch = sample_batch(data, key, 8)
            grads = jax.grad(loss)(params, batch, task)['params']
            self.assertGreater(float(jnp.linalg.norm(grads['Dense_0']['kernel'])), 0)
            self.assertGreater(float(jnp.linalg.norm(grads['AuxPrediction']['kernel'])), 0)
            for i in range(3, 8):
                for gradient in jax.tree_util.tree_leaves(grads[f'Dense_{i}']):
                    np.testing.assert_array_equal(gradient, np.zeros_like(gradient))
            invalid = batch[:-1] + (jnp.zeros_like(batch[-1]),)
            self.assertEqual(float(loss(params, invalid, task)), 0)


if __name__ == '__main__':
    unittest.main()

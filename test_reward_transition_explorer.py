"""Meaningful isolation, clustering, state decoding and HTTP checks for the atlas."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

import numpy as np

from concept_mapping.reward_transition_explorer.build import (
    assignment, describe_state, eligible, fit_kmeans, normalize, validate_bank)
from concept_mapping.reward_transition_explorer.serve import Experiment, handler_for


class ExplorerTests(unittest.TestCase):
    def test_kmeans_recovers_geometry_and_is_reproducible(self):
        x = normalize(np.repeat(np.array([[1., 0.], [0., 1.], [-1., 0.]], np.float32), 20, axis=0))
        a = fit_kmeans(x, 3, 50)
        b = fit_kmeans(x, 3, 50)
        np.testing.assert_array_equal(a, b)
        groups = assignment(x, a).reshape(3, 20)
        self.assertEqual(len(np.unique(groups[:, 0])), 3)
        for group in groups:
            self.assertTrue(np.all(group == group[0]))

    def test_reward_filter_accepts_only_plus_one(self):
        reward = np.array([1., .1, -.1, 0., .9, 1.1, 2., 1.+1e-7, 1.+1e-5])
        np.testing.assert_array_equal(eligible(reward, 'achievement_reward'), [1, 0, 0, 0, 0, 0, 0, 1, 0])
        with self.assertRaises(ValueError):
            eligible(reward, 'positive_reward')

    def test_state_decoder_preserves_mobs_inventory_and_direction(self):
        obs = np.zeros(1345, np.float32)
        obs[:1323].reshape(63, 21)[:, 2] = 1
        obs[17] = obs[19] = 1
        obs[1323] = .7
        obs[1335] = .9
        obs[1341] = 1
        obs[1343] = .5
        obs[1344] = 1
        state = describe_state(obs)
        self.assertEqual(state['blocks'], [2] * 63)
        self.assertEqual(state['mobs'][0], 5)
        self.assertEqual(state['stats'][0], 7)
        self.assertEqual(state['stats'][12], 9)
        self.assertEqual(state['direction'], 2)
        self.assertTrue(state['sleeping'])

    def fixture(self, directory, altered=False):
        source = Path(directory) / 'source'; source.mkdir()
        out = Path(directory) / 'out'; out.mkdir()
        n = 40
        bank = dict(before=np.zeros((n, 1345), np.float32), after=np.zeros((n, 1345), np.float32),
                    labels=np.tile([0, 1], n // 2), steps=np.arange(n), actions=np.zeros(n, int),
                    episode_uid=np.arange(n), split_id=np.repeat([0, 1], n // 2), rewards=np.ones(n))
        features = np.tile(np.eye(2, dtype=np.float32), (n // 2, 1))
        if altered:
            # Both evaluation labels AND evaluation features change. The fit must not.
            bank['labels'][:] = 3
            features[20:] = -features[20:]
        np.savez(source / 'transition_bank.npz', **bank)
        np.savez(source / 'features.npz', actor_delta=features)
        (out / 'manifest.json').write_text(json.dumps(dict(source=str(source), indices=list(range(n)))))
        (out / 'index.html').write_text('<!doctype html><title>test</title>')
        return Experiment(out), bank

    def test_new_fit_excludes_heldout_features_and_all_labels(self):
        params = dict(method='actor_delta', mode='achievement_reward', k=2, seed=50, cap=256, iterations=10)
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            original, bank = self.fixture(a)
            altered, _ = self.fixture(b, altered=True)
            first, second = original.fit(params), altered.fit(params)
            with np.load(original.out / 'fits' / (first['id'] + '.npz')) as x, np.load(altered.out / 'fits' / (second['id'] + '.npz')) as y:
                np.testing.assert_array_equal(x['centers'], y['centers'])
                self.assertTrue(np.all(bank['split_id'][x['fit_indices']] == 0))
            self.assertEqual(first['metrics']['mapped_accuracy'], 1.)
            self.assertEqual(first['test_n'], 20)
            self.assertEqual(first, original.fit(params))
            bank['episode_uid'][20] = bank['episode_uid'][0]
            with self.assertRaisesRegex(ValueError, 'overlap'):
                validate_bank(bank)

    def test_new_fit_excludes_health_rewards_from_fitting_and_scoring(self):
        params = dict(method='actor_delta', mode='achievement_reward', k=2, seed=50, cap=256, iterations=10)
        with tempfile.TemporaryDirectory() as tmp:
            experiment, bank = self.fixture(tmp)
            bank['rewards'][[0, 20]] = .1
            bank['rewards'][[1, 21]] = 1.1
            np.savez(experiment.source / 'transition_bank.npz', **bank)
            result = experiment.fit(params)
            self.assertEqual(result['reference_n'], 18)
            self.assertEqual(result['test_n'], 18)
            self.assertTrue(all(result['assignments'][i] == -1 for i in [0, 1, 20, 21]))
            with np.load(experiment.out / 'fits' / (result['id'] + '.npz')) as data:
                for name in ('fit_indices', 'reference_indices', 'test_indices'):
                    self.assertTrue(np.all(eligible(bank['rewards'][data[name]], 'achievement_reward')))

    def test_http_serves_atlas_rejects_other_origins_and_invalid_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            experiment, _ = self.fixture(tmp)
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(experiment))
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            url = f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(url) as response:
                    self.assertIn(b'<title>test</title>', response.read())
                with self.assertRaises(HTTPError) as denied:
                    urlopen(Request(url + '/api/fit', data=b'{}', headers={'Origin':'https://example.org'}))
                self.assertEqual(denied.exception.code, 403)
                with self.assertRaises(HTTPError) as invalid:
                    urlopen(Request(url + '/api/fit', data=b'{"k":true}'))
                self.assertEqual(invalid.exception.code, 400)
                with self.assertRaises(HTTPError) as traversal:
                    urlopen(url + '/../../source/transition_bank.npz')
                self.assertEqual(traversal.exception.code, 404)
                params = dict(method='actor_delta', mode='achievement_reward', k=2, seed=50, cap=256, iterations=10)
                request = Request(url + '/api/fit', data=json.dumps(params).encode(), headers={'Content-Type':'application/json', 'Origin':url})
                with urlopen(request) as response:
                    result = json.load(response)
                self.assertEqual(result['metrics']['mapped_accuracy'], 1.)
            finally:
                server.shutdown(); server.server_close(); thread.join()


if __name__ == '__main__':
    unittest.main()

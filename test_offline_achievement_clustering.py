import unittest
import numpy as np
from concept_mapping.test_achievement_clustering_offline import (
    sample_episode, partition_metrics, classify_clusters, fit_kmeans_unlabeled, assign_clusters)


class OfflineClusteringTests(unittest.TestCase):
    def test_sampling_does_not_use_achievement_labels(self):
        obs = np.zeros((8, 1345), np.float32)
        flags = np.zeros((8, 22), bool)
        rewards = np.array([0, 0, .1, 0, 1, 0, -.1, 0], np.float32)
        a = sample_episode(obs, flags, np.zeros(8), rewards, obs[-1], flags[-1], 2, 50)
        flags[5:, 0] = True
        b = sample_episode(obs, flags, np.zeros(8), rewards, obs[-1], flags[-1], 2, 50)
        np.testing.assert_array_equal(a['steps'], b['steps'])
        self.assertTrue(set((2, 4, 6)).issubset(a['steps']))
        self.assertNotEqual(a['labels'].tolist(), b['labels'].tolist())

    def test_terminal_health_event_uses_true_post_state(self):
        obs = np.zeros((2, 1345), np.float32); obs[:, 1335] = .5
        flags = np.zeros((2, 22), bool)
        terminal = obs[-1].copy(); terminal[1335] = .4
        result = sample_episode(obs, flags, np.zeros(2), np.array([0, -.1]), terminal, flags[-1], 2, 50)
        self.assertEqual(int(result['labels'][-1]), 23)
        self.assertAlmostEqual(float(result['after'][-1, 1335]), .4)

    def test_metrics_and_mapping_are_permutation_invariant(self):
        labels = np.array([0, 0, 1, 1]); clusters = np.array([1, 1, 0, 0])
        metrics = classify_clusters(labels, clusters, labels, clusters, 2)[0]
        self.assertEqual(metrics['mapped_accuracy'], 1)
        self.assertEqual(metrics['achievement_macro_f1'], 1)
        p = partition_metrics(np.array([[0, 2], [2, 0]]))
        self.assertAlmostEqual(p['ari'], 1); self.assertAlmostEqual(p['nmi'], 1)
        mixed = partition_metrics(np.array([[1, 1], [1, 1]]))
        self.assertAlmostEqual(mixed['nmi'], 0)

    def test_unlabeled_fit_recovers_distinct_synthetic_groups(self):
        x = np.repeat(np.array([[1., 0.], [0., 1.], [-1., 0.]], np.float32), 8, axis=0)
        centers = fit_kmeans_unlabeled(x, 3, 50, iterations=5)
        assignments = assign_clusters(x, centers)
        result = classify_clusters(np.repeat(np.arange(3), 8), assignments,
                                   np.repeat(np.arange(3), 8), assignments, 3)[0]
        self.assertAlmostEqual(result['ari'], 1.)


if __name__ == '__main__':
    unittest.main()

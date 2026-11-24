import unittest
import tempfile
from pathlib import Path
import numpy as np
from concept_mapping.compare_transport_achievement_groups import (
    achievement_reward_proxy, match_events, nearest_prototypes,
    select_anchor, audit_classifier, build_transport_groups, GROUPINGS, resolve_encoder_metadata,
    bootstrap_metrics)


class TransportGroupsTests(unittest.TestCase):
    def test_bootstrap_macro_excludes_unobserved_classes(self):
        labels = np.array([0, 1, 0, 1])
        intervals, macro, accuracy = bootstrap_metrics(labels, labels, np.array([1, 1, 2, 2]), samples=30)
        np.testing.assert_array_equal(macro, np.ones(30))
        self.assertEqual(intervals['macro_f1_95_ci'], [1., 1.])

    def test_own_policy_split_and_quotas(self):
        from concept_mapping.group_own_policy_achievement_events import episode_quotas, split_episodes
        quotas, offsets = episode_quotas(1000, 32)
        ids = np.concatenate([np.arange(start, start+count) for start, count in zip(offsets, quotas)])
        np.testing.assert_array_equal(ids, np.arange(1000))
        split = split_episodes(1000, 4050)
        self.assertEqual(int((split == 0).sum()), 700)
        np.testing.assert_array_equal(split, split_episodes(1000, 4050))

    def test_encoder_override_does_not_mutate_collection_metadata(self):
        collection = dict(run_path='collection', checkpoint=9999941632, seed=50)
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory); (run/'files/policies/49995776').mkdir(parents=True)
            encoder = resolve_encoder_metadata(collection, run, 49995776)
            self.assertEqual(encoder['checkpoint'], 49995776)
            self.assertEqual(collection['checkpoint'], 9999941632)
            self.assertEqual(encoder['seed'], 50)
        with self.assertRaises(ValueError):
            resolve_encoder_metadata(collection, 'run', None)

    def test_grouping_sweep_includes_32_without_removing_existing_conditions(self):
        self.assertEqual(GROUPINGS, ('transport_anchor', 'kmeans22', 'kmeans32', 'kmeans64', 'kmeans128'))
        self.assertEqual(int('kmeans32'.replace('kmeans', '')), 32)

    def test_reward_detection_handles_coincident_health_loss(self):
        before = np.zeros((3, 1345), np.float32); after = before.copy()
        before[:, 1335] = .9; after[:, 1335] = [.3, .8, .9]
        reward = [.4, -.1, 1.]
        np.testing.assert_allclose(achievement_reward_proxy(before, after, reward), [1, 0, 1], atol=1e-6)

    def test_transport_permutation_and_missing_event(self):
        a = np.eye(3, dtype=np.float32)
        b = a[[2, 0]]
        groups, _, residual, _ = match_events(a, b)
        np.testing.assert_array_equal(groups, [1, -1, 0])
        self.assertLess(residual, 1e-6)

    def test_anchor_uses_event_counts_and_group_audit_handles_abstentions(self):
        anchor, uid = select_anchor(np.array([10, 10, 11, 11, 11]), 50)
        self.assertEqual(uid, 11); self.assertEqual(len(anchor), 3)
        result = audit_classifier(np.array([0, 1, 0]), np.array([1, 0, -1]),
            np.array([0, 1, 0]), np.array([1, 0, -1]), 2)
        self.assertAlmostEqual(result[0]['accuracy'], 2/3)
        self.assertEqual(int(result[6][-1]), 26)

    def test_unlabeled_transport_groups_predict_new_events(self):
        x = np.vstack([np.eye(3), np.eye(3)[[2, 0]]]).astype(np.float32)
        uids = np.array([10, 10, 10, 11, 11])
        centers, groups, _, _, _, _, _ = build_transport_groups(x, uids, 50, 1)
        predicted = nearest_prototypes(np.eye(3), centers)
        np.testing.assert_array_equal(predicted, groups[:3])

    def test_raw_conflicting_edges_are_preserved_but_groups_resolved(self):
        def unconverged(*args, **kwargs):
            return np.array([[.6, .6], [0., 0.]])
        groups, _, residual, audit = match_events(np.eye(2), np.eye(2), unconverged)
        np.testing.assert_array_equal(groups, [0, -1])
        self.assertTrue(audit['conflict']); self.assertEqual(len(audit['raw_source']), 2)
        self.assertAlmostEqual(residual, .2)


if __name__ == '__main__':
    unittest.main()

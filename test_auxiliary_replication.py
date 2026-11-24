import json
from pathlib import Path
import tempfile
import unittest

from aux_search.recipes import BASELINE
from aux_search.replication import FROZEN, GLOBAL_CAP, SEEDS, Replication, paired_summary, prepare


class ReplicationTests(unittest.TestCase):
    def source(self, root):
        source = Path(root) / 'source'
        source.mkdir()
        (source / 'study.json').write_text(json.dumps(dict(state='complete', trials=[dict(seed=50)])))
        (source / 'results.json').write_text(json.dumps(dict(interactions_charged=172186112,
            confirmation={FROZEN.id:dict(recipe=FROZEN.to_dict())})))
        return source

    def test_protocol_is_frozen_and_reserves_all_pairs_within_original_cap(self):
        with tempfile.TemporaryDirectory() as root:
            source, out = self.source(root), Path(root) / 'replication'
            protocol = prepare(out, source)
            self.assertEqual(protocol, prepare(out, source))
            self.assertEqual(protocol['training_seeds'], list(range(250, 258)))
            self.assertEqual(protocol['evaluation_steps_per_trial'], 112640)
            self.assertLessEqual(protocol['prior_interactions_charged'] + 16000000 +
                                 protocol['evaluation_reserve'], GLOBAL_CAP)
            data = json.loads((out / 'study.json').read_text())
            data['state'] = 'running'
            (out / 'study.json').write_text(json.dumps(data))
            prepare(out, source)
            self.assertEqual(json.loads((out / 'study.json').read_text())['state'], 'running')
            (source / 'study.json').write_text(json.dumps(dict(state='complete', trials=[dict(seed=250)])))
            with self.assertRaisesRegex(ValueError, 'overlap'):
                prepare(out, source)

    def test_uncertainty_uses_seed_pairs_and_rejects_nonfinite_or_unmatched_returns(self):
        result = paired_summary([1.] * 8, [1., 3.] * 4)
        self.assertEqual(result['mean_difference'], 1.)
        self.assertAlmostEqual(result['paired_standard_error'], 1 / (7 ** .5))
        self.assertAlmostEqual(result['difference_ci95'][0], .10625604)
        self.assertAlmostEqual(result['difference_ci95'][1], 1.89374396)
        self.assertEqual(result['positive_seed_pairs'], 4)
        self.assertEqual(result['conclusion'], 'positive')
        self.assertEqual(paired_summary([1.] * 8, [0., 2.] * 4)['conclusion'], 'inconclusive')
        for baseline, auxiliary in [([1], [1]), ([1, 2], [1]), ([1, 2], [1, float('nan')])]:
            with self.assertRaises(ValueError):
                paired_summary(baseline, auxiliary)

    def test_report_requires_every_pair_and_checks_frozen_trial_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            source, out = self.source(root), Path(root) / 'replication'
            study = Replication(out, source, [0])
            for seed in SEEDS:
                for recipe in (BASELINE, FROZEN):
                    key = f'{recipe.id}_seed{seed}'
                    directory = out / 'trials' / key
                    directory.mkdir(parents=True)
                    result = dict(complete=True, training_seed=seed, training_steps=999424,
                        checkpoint=999424, episodes=256, recipe=recipe.to_dict(),
                        evaluation_seed=study.protocol['evaluation_seeds'][str(seed)],
                        mean_return=5. + (recipe == FROZEN) * .25, environment_steps=61440,
                        achievements={str(i):10. for i in range(22)})
                    (directory / 'result.json').write_text(json.dumps(result))
                    study.data['trials'].append(dict(id=key, recipe_id=recipe.id, seed=seed,
                        state='complete', training_reserved=1000000, evaluation_reserved=112640,
                        evaluation_steps=61440, confirmation=True))
            report = study.report()
            self.assertEqual(report['primary']['seed_pairs'], 8)
            self.assertEqual(report['primary']['mean_difference'], .25)
            self.assertFalse(report['old_search_or_confirmation_seeds_pooled'])
            self.assertEqual(report['combined_interactions_charged'], 172186112 + 16000000 + 16 * 61440)
            study.data['trials'][-1]['state'] = 'running'
            with self.assertRaisesRegex(RuntimeError, 'All predeclared'):
                study.report()
            study.data['trials'][-1]['state'] = 'complete'
            path = directory / 'result.json'
            result['evaluation_seed'] = 3050
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(RuntimeError, 'frozen replication protocol'):
                study.report()


if __name__ == '__main__':
    unittest.main()

"""Deferred continuations must restore the final save, never an older milestone."""
import json,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from ppo_experiment_config import Experiment,Architecture,PPO,Budget,Evaluation
from run_ppo_study import resolve_queued_resume,prepare,queue


class ContinuationQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'source';self.source.mkdir()
        self.progress=self.source/'progress.json'
        self.progress.write_text(json.dumps(dict(state='training',latest_checkpoint=65536)))
        self.spec=Experiment('queued',resume_after=str(self.source),
            architectures=(Architecture('spatial_map_gru',1,5),),
            ppo=PPO(lr_horizon_timesteps=10000000000),
            budget=Budget(max_hours=48,max_steps=2000027648),
            evaluation=Evaluation(common_checkpoints={'1b':1000000000}))
        self.spec.validate()

    def final_save(self,**changes):
        checkpoint=self.root/'131072';checkpoint.mkdir(exist_ok=True)
        (checkpoint/'_METADATA').touch()
        data=dict(state='complete',training_exit_code=0,final_checkpoint=str(checkpoint),latest_checkpoint=65536)
        data.update(changes);self.progress.write_text(json.dumps(data));return checkpoint

    def test_waits_until_source_segment_finishes(self):
        self.assertIsNone(resolve_queued_resume(self.spec))
        self.final_save(state='evaluating_final')
        self.assertIsNone(resolve_queued_resume(self.spec))

    def test_resolves_actual_final_not_stale_latest_and_preserves_training(self):
        checkpoint=self.final_save()
        data=resolve_queued_resume(self.spec)
        config=self.root/'resolved.json';config.write_text(json.dumps(data))
        resolved=Experiment.load(config)
        self.assertEqual(resolved.resume_from,str(checkpoint));self.assertIsNone(resolved.resume_after)
        self.assertFalse(resolved.resume_reset_optimizer)
        self.assertEqual(resolved.ppo,self.spec.ppo);self.assertEqual(resolved.seed,50)
        self.assertEqual(resolved.rewards,self.spec.rewards)
        self.assertEqual(resolved.budget,self.spec.budget)
        self.assertTrue(all(step>int(checkpoint.name) for step in resolved.checkpoint_milestones))
        self.assertIn(int(checkpoint.name)+1000000,resolved.checkpoint_milestones)
        metadata=resolved.metadata(resolved.architectures[0],self.root,'/python')
        self.assertEqual(metadata['start_timesteps'],int(checkpoint.name))
        self.assertEqual(metadata['learning_rate_horizon_steps'],10000000000)

    def test_failed_or_incomplete_source_never_falls_back(self):
        for changes in ({'training_exit_code':1},{'final_checkpoint':None},{'state':'failed'}):
            with self.subTest(changes=changes):
                self.final_save(**changes)
                with self.assertRaises(RuntimeError):resolve_queued_resume(self.spec)
        checkpoint=self.final_save();(checkpoint/'_METADATA').unlink()
        with self.assertRaisesRegex(RuntimeError,'incomplete'):resolve_queued_resume(self.spec)

    def test_source_already_at_target_is_rejected(self):
        self.final_save(final_checkpoint=str(self.root/'2000027648'))
        checkpoint=self.root/'2000027648';checkpoint.mkdir();(checkpoint/'_METADATA').touch()
        with self.assertRaisesRegex(ValueError,'already meets'):resolve_queued_resume(self.spec)

    def test_ambiguous_queue_configuration_and_direct_preparation_rejected(self):
        for changes in ({'resume_from':'/tmp/65536'},{'resume_reset_optimizer':True},
                        {'architectures':(Architecture('mlp',0,4),Architecture('gru',1,5))}):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError,'Queued resume'):replace(self.spec,**changes).validate()
        with self.assertRaisesRegex(ValueError,'--queue'):prepare(self.spec,self.root/'study')

    def test_duplicate_queue_cannot_launch_another_worker(self):
        study=self.root/'study';study.mkdir();(study/'queue_manifest.json').write_text('{}')
        with self.assertRaises(FileExistsError):queue(self.spec,study)

    def test_explicit_checkpoint_can_wait_for_devices(self):
        checkpoint=self.final_save()
        spec=replace(self.spec,resume_after=None,resume_from=str(checkpoint))
        self.assertEqual(resolve_queued_resume(spec),spec.to_dict())
        (checkpoint/'_METADATA').unlink()
        with self.assertRaisesRegex(RuntimeError,'incomplete'):resolve_queued_resume(spec)


if __name__=='__main__':unittest.main()

"""Multiple inference temperatures and shared evaluation GPU serialization."""
import unittest
from cap_reward_shaped_runs import evaluation_jobs,try_evaluation_lock,pending_common_evaluations


class EvaluationSchedulingTests(unittest.TestCase):
    def test_deferred_checkpoints_are_evaluated_in_order_after_training(self):
        targets=[('10m',10,1.),('25m',25,1.),('50m',50,1.)]
        available={10:'first',25:'second',100:'final'}
        self.assertEqual(pending_common_evaluations(targets,set(),available),
                         [('10m','first',1.),('25m','second',1.)])
        self.assertEqual(pending_common_evaluations(targets,{'10m'},available),[('25m','second',1.)])

    def test_default_and_multiple_temperatures(self):
        targets,final=evaluation_jobs({'common_checkpoints':[['early',65536]]})
        self.assertEqual(targets,[('early',65536,1.)]);self.assertEqual(final,[('wallclock',1.)])
        targets,final=evaluation_jobs({'common_checkpoints':[['early',65536]],'temperatures':[1.,.25]})
        self.assertEqual(targets,[('early',65536,1.),('early_t0p25',65536,.25)])
        self.assertEqual(final,[('wallclock',1.),('wallclock_t0p25',.25)])

    def test_gpu_lock_is_exclusive_until_released(self):
        # A dedicated synthetic ID tests the lock protocol without touching GPUs.
        first=try_evaluation_lock(987654)
        try:
            self.assertIsNotNone(first)
            second=try_evaluation_lock(987654)
            self.assertIsNone(second)
        finally:first.close()
        third=try_evaluation_lock(987654)
        self.assertIsNotNone(third);third.close()


if __name__=='__main__':unittest.main()

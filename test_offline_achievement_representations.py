import unittest
import numpy as np
from concept_mapping.test_achievement_representations_offline import extract_events,select_reference,retrieval_scores


class OfflineAchievementTests(unittest.TestCase):
    def test_terminal_event_uses_real_final_observation(self):
        obs=np.array([[0.,0.],[1.,0.],[2.,0.]],np.float32)
        flags=np.zeros((3,22),bool);flags[1:,0]=True
        final=flags[-1].copy();final[1]=True
        data=extract_events(obs,flags,np.array([5,1,5]),np.array([9.,9.]),final)
        np.testing.assert_array_equal(data['steps'],[0,2])
        np.testing.assert_array_equal(data['labels'],[0,1])
        np.testing.assert_array_equal(data['after'],[[1.,0.],[9.,9.]])
        self.assertEqual(data['terminal_events'],1)

    def test_excludes_compound_and_rejects_flag_regression(self):
        obs=np.ones((2,3),np.float32);flags=np.zeros((2,22),bool);flags[1,:2]=True
        data=extract_events(obs,flags,np.array([5,5]),obs[-1],flags[-1])
        self.assertEqual(data['compound_events'],1);self.assertEqual(len(data['labels']),0)
        with self.assertRaises(ValueError):extract_events(obs,flags,np.array([5,5]),obs[-1],np.zeros(22,bool))

    def test_reference_excludes_test_and_ties_are_not_order_dependent(self):
        bank={'labels':np.array([0,1,0,0]),'split_id':np.array([0,0,1,0])}
        reference=select_reference(bank['labels'],bank['split_id'],1,50)
        self.assertTrue((bank['split_id'][reference]==0).all())
        feature=np.array([[0.,0.],[0.,0.],[0.,0.],[1.,0.]],np.float32)
        scores,_,_,ties,zero=retrieval_scores(feature,bank,np.array([0,1,3]),np.array([2]),1)
        self.assertAlmostEqual(float(scores[0]),.5);self.assertEqual(int(ties[0]),2);self.assertEqual(zero,1.)


if __name__=='__main__':unittest.main()

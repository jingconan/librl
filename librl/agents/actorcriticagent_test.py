from __future__ import print_function, division, absolute_import
import unittest
import scipy
from numpy.testing import assert_array_almost_equal
from .actorcriticagent import ActorCriticAgent
from ..testutil import MockPolicy, MockLearner

class TDLearnerTestCase(unittest.TestCase):
    def setUp(self):
        self.policy = MockPolicy({
            'obs_1': 'action_1',
        })

        self.learner = MockLearner(self.policy)
        self.agent = ActorCriticAgent(self.learner, 1, 1)

    def testGetAction(self):
        self.agent.lastobs = 'obs_1'
        self.assertEqual('action_1', self.agent.getAction())

if __name__ == "__main__":
    unittest.main()

from __future__ import print_function, division, absolute_import
from .td import TDLearner

import unittest
import scipy
from numpy.testing import assert_array_almost_equal

from pybrain.datasets import ReinforcementDataSet
from librl.policies.boltzmann import BoltzmanPolicy

class MockTDLearnerForTest(TDLearner):
    def actor(self, lastobs, lastaction, lastfeature):
        pass

class TDLearnerTestCase(unittest.TestCase):
    def setUp(self):
        self.theta = [0.4, 1.1]
        self.policy = BoltzmanPolicy(4, 2, self.theta)

        self.dataset = ReinforcementDataSet(8, 1)
        feature1 = scipy.array([
            (0.6, 0.2),
            (0.3, 0.6),
            (0.4, 0.01),
            (0.5, -0.2)
        ])
        #  feature2 = scipy.array([
        #      (0.3, 0.6),
        #      (0.6, 0.2),
        #      (50, -20),
        #      (0.4, 0.01),
        #  ])
        #  feature3 = scipy.array([
        #      (0.1, 0.1),
        #      (0.2, 0.2),
        #      (0.3, -0.3),
        #      (0.4, 0.4),
        #  ])

        self.dataset.addSample(feature1.reshape(-1), 0, 0)
        self.dataset.addSample(feature1.reshape(-1), 1, 1)
        self.dataset.addSample(feature1.reshape(-1), 2, 1.5)
        self.dataset.addSample(feature1.reshape(-1), 3, 0.5)

    # See https://goo.gl/7VMeDS for the spreadsheet that checks the math.
    # Note that actor is disabled in this test.
    def testLearnOnDataSet(self):
        learner = MockTDLearnerForTest(self.policy,
                            tracestepsize=0.9,
                            actorstepsize= 1,
                            maxcriticnorm=1)
        #  print('theta before: ', self.policy.theta)
        learner.learnOnDataSet(self.dataset)
        #  print('theta after: ', self.policy.theta)
        assert_array_almost_equal([0.75], learner.alpha)
        assert_array_almost_equal([0.6502086349], learner.d)
        assert_array_almost_equal([-0.05134717282,
                                   0.0955041449,
                                   -0.006562892692,
                                   0.009694943872,
                                   0.009694943872,
                                   -0.04286319107], learner.r)
        assert_array_almost_equal([-0.0952710795,
                                   -0.2401405293,
                                   -0.0374024173,
                                   0.0552522117,
                                   0.0552522117,
                                   -0.2442805382], learner.z)
    def testActor(self):
        learner = TDLearner(self.policy,
                            tracestepsize=0.9,
                            actorstepsize= 1,
                            maxcriticnorm=1000)
        learner.r = scipy.array([1, 1, 1, 1, 1, 1], dtype=float)
        lastfeature = scipy.array([1, 2, 3, 4, 5, 6])
        learner.actor([], [], lastfeature)
        # the stateActionValue = 1*1 + 1*2 + 1*3 + 1*4 + ... + 1*7 = 21.
        # the initial theta is [0.4, 1.1], the update is [1, 2] * 21.
        assert_array_almost_equal([21.4, 43.1], learner.module.theta)

if __name__ == "__main__":
    unittest.main()



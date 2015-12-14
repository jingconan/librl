from __future__ import print_function, division, absolute_import
from .hessian import HessianLearner
from ..testutil import MockPolicy, MockPolicyFeatureModule

import unittest
import scipy
from numpy.testing import assert_array_almost_equal, assert_almost_equal

from pybrain.datasets import ReinforcementDataSet
from librl.policies.boltzmann import BoltzmanPolicy

class TDLearnerTestCase(unittest.TestCase):
    def setUp(self):
        self.policy = MockPolicy({}, [0, 0])
        self.module = MockPolicyFeatureModule(self.policy)
        self.module.outdim = 6
        self.tracestepsize = 0.9
        self.actorstepsize = 3
        self.maxcriticnorm = 1000
        self.hessianlearningrate = 0.9
        self.learner = HessianLearner(self.hessianlearningrate, self.policy, self.tracestepsize,
                                      self.actorstepsize, self.maxcriticnorm,
                                      self.module)

    def testActor(self):
        self.learner.H = scipy.array([[2, 0], [0, 3]])
        self.learner.scaledfeature = scipy.array([4, 5])
        self.learner.hessiansamplenumber = 1
        self.learner.actor([], [], [])
        assert_array_almost_equal([6, 10], self.learner.module.theta)

    def testCritic(self):
        #  self.module.outdim = 6
        self.learner.A = scipy.eye(6)
        self.learner.b = -1 * scipy.ones((6,))
        self.learner.V = scipy.array([[1,  0],
                                      [0, -1],
                                      [1,  0],
                                      [0, -1],
                                      [1,  0],
                                      [1,  0]])
        self.learner.r = scipy.array([1, 1, 1, 1, 1, 1])

        lastfeature = scipy.array([-1, -1, -1, -1, -1, -1])
        lastreward = 1
        feature = scipy.array([1, 2, 3, 4, 5, 6])
        reward = 1
        self.learner.critic(lastreward, lastfeature, reward, feature)

        assert_array_almost_equal([[-1, 0],
                                   [ 0, 1],
                                   [-1, 0],
                                   [ 0, 1],
                                   [-1, 0],
                                   [-1, 0]], self.learner.T)
        expectedu = [[12, 18],
                     [39, 66]]
        assert_array_almost_equal(expectedu, self.learner.U)
        assert_array_almost_equal(scipy.linalg.inv(expectedu), self.learner.H)
        assert_array_almost_equal([21, 42], self.learner.eta)

from __future__ import print_function, division, absolute_import
import scipy
import tempfile
import unittest
from numpy.testing import assert_array_almost_equal

from .garnet import GarnetEnvironment, GarnetTask
from librl.util import zdump

class GarnetEnvironmentTestCase(unittest.TestCase):
    def setUp(self):
        scipy.random.seed(0)
        self.testDir = tempfile.mkdtemp()

        self.numStates = 3
        self.numActions = 2
        self.branching = 1
        self.feaDim = 5
        self.feaSum = 2

        # For action 0
        # state 0 -> 1
        # state 1 -> 0
        # state 2 -> 2
        ts0 = [[1], [0], [2]]
        tp0 = [[1], [1], [1]]
        # For action 1
        # state 0 -> 2
        # state 1 -> 1
        # state 2 -> 0
        ts1 = [[2], [1], [0]]
        tp1 = [[1], [1], [1]]
        self.transitionStates = scipy.array([ts0, ts1])
        self.transitionProb = scipy.array([tp0, tp1],
                                          dtype=float)
        self.stateObs = [(1, 1, 0, 0, 0),
                         (1, 0, 1, 0, 0),
                         (0, 1, 0, 1, 0)]

        message = {
            'transitionStates': self.transitionStates,
            'transitionProb': self.transitionProb,
            'stateObs': self.stateObs,
        }
        self.loadPath = self.testDir + 'transition_model.pkz'
        zdump(message, self.loadPath)

        self.env = GarnetEnvironment(numStates=self.numStates,
                                     numActions=self.numActions,
                                     branching=self.branching,
                                     feaDim=self.feaDim,
                                     feaSum=self.feaSum,
                                     loadPath=self.loadPath)

    def testGenStateObs(self):
        self.env._genStateObs()
        self.assertEqual(self.numStates, len(set(self.stateObs)))
        p = scipy.sum(scipy.array(self.env.stateObs), axis=1)
        expected = self.feaSum * scipy.ones((self.numStates,))
        assert_array_almost_equal(expected, p)

    def testGenTransitionTable(self):
        branching = 2
        self.env.branching = branching
        self.env._genTransitionTable()
        # Assert the the shape of generated transitionStates and transitionProb.
        assert_array_almost_equal([self.numActions, self.numStates,
                                   branching],
                                  self.env.transitionStates.shape)
        assert_array_almost_equal([self.numActions, self.numStates,
                                   branching],
                                  self.env.transitionProb.shape)
        assert_array_almost_equal(scipy.ones((self.numActions,
                                              self.numStates)),
                                  scipy.sum(self.env.transitionProb, axis=2))

    def testGetSensors(self):
        self.env.curState = 0
        assert_array_almost_equal(self.stateObs[0], self.env.getSensors())
        self.env.curState = 1
        assert_array_almost_equal(self.stateObs[1], self.env.getSensors())
        self.env.curState = 2
        assert_array_almost_equal(self.stateObs[2], self.env.getSensors())

    def testPerformAction(self):
        # test for action 0.
        self.env.curState = 0
        self.env.performAction([0])
        # transit from 0 to 1
        self.assertEqual(1, self.env.curState)
        self.env.performAction([0])
        # transit from 1 to 0
        self.assertEqual(0, self.env.curState)
        self.env.curState = 2
        self.env.performAction([0])
        # stay at 2
        self.assertEqual(2, self.env.curState)

        # test for action 1.
        self.env.curState = 0
        self.env.performAction([1])
        # transit from 0 to 2
        self.assertEqual(2, self.env.curState)
        self.env.performAction([1])
        # transit from 2 to 0
        self.assertEqual(0, self.env.curState)
        self.env.curState = 1
        # stay at 1
        self.env.performAction([1])
        self.assertEqual(1, self.env.curState)


class MockGarnetEnvironment(object):
    def __init__(self, numStates, numActions, curState, prevState,
                 lastAction, sensors):
        self.numStates = numStates
        self.numActions = numActions
        self.curState = curState
        self.prevState = prevState
        self.lastAction = lastAction
        self.sensors = sensors

    def getSensors(self):
        return self.sensors

    def performAction(self, action):
        self.lastAction = action

class GarnetTaskTestCase(unittest.TestCase):
    def setUp(self):
        self.env = MockGarnetEnvironment(3, 2, 0, 1,
                                         scipy.array([0]),
                                         scipy.array([1, 1, 0,
                                                      0, 0],
                                                     dtype=float))
        self.task = GarnetTask(self.env, sigma=1)

    def testGetObservation(self):
        expectedFeature = [[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]]
        expected = scipy.array(expectedFeature).reshape(-1)
        assert_array_almost_equal(expected, self.task.getObservation())

    def testGetReward(self):
        scipy.random.seed(0)
        # with seed 0, the first run of randn will return 1.764052345967664,
        # which is the expected reward. The second run of randn will return
        # 0.4001572083672233. So the reward is:
        # reward = 1 * 0.4001572083672233 + 1.764052345967664
        self.assertAlmostEqual(2.1642095543348874, self.task.getReward())

if __name__ == '__main__':
    unittest.main()

import unittest
import scipy
from .robottask import RobotMotionAvgRewardTask
from ..trapmaze import TrapMaze
from numpy.testing import assert_array_almost_equal

class RobotMotionTaskTestCase(unittest.TestCase):
    def setUp(self):

        self.theta = [0.4, 1.1]

        self.topology = [[0, 0, 0],
                         [0, -1, 0],
                         [2, 0, 2]]

        self.tranProb = [
            [0.7, 0.1, 0.1, 0.1], # N
            [0, 0.8, 0.1, 0.1], # E
            [0, 0.1, 0.9, 0], # S
            [0, 0.1, 0.1, 0.8], # W
        ]

        self.maze = TrapMaze(self.topology, (0, 0), self.tranProb)
        self.task = RobotMotionAvgRewardTask(self.maze, senseRange=1)

    # See https://goo.gl/fmRKbV for spreadsheet that checks this case.
    def testGetObservation(self):
        expected = [0, 0, 0, -0.777777778, 0, 0.8, 0, 0]
        assert_array_almost_equal(expected, self.task.getObservation())

    def testGetReward(self):
        self.assertEqual(self.task.DEFAULT_REWARD, self.task.getReward())

        maze2 = TrapMaze(self.topology, (1, 1), self.tranProb)
        task2 = RobotMotionAvgRewardTask(maze2, senseRange=1)
        self.assertEqual(task2.TRAP_REWARD, task2.getReward())

        maze3 = TrapMaze(self.topology, (2, 0), self.tranProb)
        task3 = RobotMotionAvgRewardTask(maze3, senseRange=1)
        self.assertEqual(task3.GOAL_REWARD, task3.getReward())

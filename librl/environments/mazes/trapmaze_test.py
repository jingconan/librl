from __future__ import print_function, division, absolute_import
from .trapmaze import TrapMaze

import unittest
import scipy
from numpy.testing import assert_array_almost_equal


class TrapMazeTestCase(unittest.TestCase):
    def setUp(self):
        self.theta = [0.4, 1.1]

        self.topology = [[1, 0, 0],
                         [1, -1, 0],
                         [1, 0, 0]]

        self.tranProb = [
            [1, 0, 0, 0], # N
            [0, 1, 0, 0], # E
            [0, 0, 1, 0], # S
            [0, 0, 0, 1], # W
        ]

        self.maze = TrapMaze(self.topology, (0, 1), self.tranProb)

    def testHitWall(self):
        self.maze.performAction(3)
        self.assertTrue(self.maze.bang)
        assert_array_almost_equal((0, 1), self.maze.perseus)

    def testMoveAndHitTrap(self):
        # move east.
        self.maze.performAction(1)
        self.assertFalse(self.maze.bang)
        assert_array_almost_equal((0, 2), self.maze.perseus)
        # move south.
        self.maze.performAction(2)
        self.assertFalse(self.maze.bang)
        assert_array_almost_equal((1, 2), self.maze.perseus)
        # hit trap and go back to start position.
        self.maze.performAction(3)
        self.assertTrue(self.maze.bang)
        assert_array_almost_equal((0, 1), self.maze.perseus)

    def testMoveOutbound(self):
        # move east.
        self.maze.performAction(1)
        self.assertFalse(self.maze.bang)
        assert_array_almost_equal((0, 2), self.maze.perseus)
        # try to move east again but will not move as it has reached the
        # boundary.
        self.maze.performAction(1)
        self.assertTrue(self.maze.bang)
        assert_array_almost_equal((0, 2), self.maze.perseus)

    def testRest(self):
        # move east.
        self.maze.performAction(1)
        self.assertFalse(self.maze.bang)
        assert_array_almost_equal((0, 2), self.maze.perseus)
        # reset
        self.maze.reset()
        self.assertFalse(self.maze.bang)
        assert_array_almost_equal((0, 1), self.maze.perseus)


if __name__ == '__main__':
    unittest.main()

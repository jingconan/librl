import unittest
from scipy import zeros
class RobotMotionTaskTestCase(unittest.TestCase):
    def setUp(self):
        print "In method", self._testMethodName
        from TrapMaze import TrapMaze
        iniState = (0, 0)
        gridSize = (5, 5)
        goalStates = [ (4, 4) ]
        unsafeStates = [ [2, 0], [3, 2], [2, 3] ]

# TP is transition probability
# The order is ENWS
        TP = [[0.7, 0.1, 0.1, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.1, 0.7, 0.1],
                [0.1, 0.1, 0.1, 0.7]]

        DF = lambda x, y: abs(x[0] - y[0]) + abs(x[1] - y[1])
        envMatrix = zeros(gridSize)
        envMatrix[zip(*unsafeStates)] = -1
        env = TrapMaze(envMatrix, iniState, goalStates, TP, DF)
        self.task = RobotMotionTask(env,
                senrange = 2)

    def test_GetAllowNSTP(self):
        pass

    def test_GetNSS(self):
        pass

    def test_getFeatureList(self):
        pass

    def test_getObservation(self):
        pass


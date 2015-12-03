from __future__ import print_function, division, absolute_import
from .boltzmann import BoltzmanPolicy

import unittest
import scipy
from numpy.testing import assert_array_almost_equal


class BoltzmanPolicyTestCase(unittest.TestCase):
    """Please see https://goo.gl/xBM8AZ for the spreadsheet of the unittest."""
    def setUp(self):
        print("In method", self._testMethodName)
        self.theta = [0.4, 1.1]
        self.policy = BoltzmanPolicy(actionnum=4,
                                     T=2,
                                     iniTheta=self.theta)
        self.features = scipy.array([
            (0.6, 0.2),
            (0.3, 0.6),
            (0.4, 0.01),
            (50, -20)
        ])

    def testActivate(self):
        scipy.random.seed(0)
        obs = self.policy.fea2obs(self.features)
        result = self.policy.activate(obs)
        assert_array_almost_equal([1], result)

    def testCalBasisFuncVal(self):
        expected = [[-4.176832772, 1.680848961],
                    [-4.476832772, 2.080848961],
                    [-4.376832772, 1.490848961],
                    [45.22316723, -18.51915104]]
        assert_array_almost_equal(expected,
                                  self.policy.calBasisFuncVal(self.features))

    def testCalSecondBasisFuncVal(self):
        expected = [[-196.7191887, 80.56815369],
                    [80.56815369, -33.04289481]]
        result = self.policy.calSecondBasisFuncVal(self.features)
        assert_array_almost_equal(expected, result)

    def testGetActionProbFirst(self):
        result = self.policy._getActionProb(self.features, self.theta)
        expected = scipy.array([0.3001868638, 0.352272548, 0.2597981959,
                                0.08774239221])
        assert_array_almost_equal(expected, result)

    def testGetActionProbSecond(self):
        for v in scipy.arange(-10, 10, 2):
            features = [
                    (v, 0.0),
                    (v, 0.3),
                    (v, 0.0),
                    (v, -0.3)
                    ]
            theta = scipy.array([10, 10]).reshape(-1, 1)
            ap = self.policy._getActionProb(features, theta)
            self.assertTrue(ap[1] > ap[3])
            self.assertEqual(ap[0], ap[2])


if __name__ == "__main__":
    unittest.main()


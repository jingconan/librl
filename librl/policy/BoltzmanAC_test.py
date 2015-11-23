from __future__ import print_function, division, absolute_import
from .BoltzmanAC import BoltzmanPolicy

import unittest
import scipy
import numpy
from numpy.testing import assert_array_almost_equal


class BoltzmanPolicyTestCase(unittest.TestCase):
    """Please see https://goo.gl/xBM8AZ for the spreadsheet of the unittest."""
    def setUp(self):
        print("In method", self._testMethodName)
        self.policy = BoltzmanPolicy(feaDim=2,
                                     numActions=4,
                                     T=2,
                                     iniTheta=[0.4, 1.1])
        self.feaList = [
                (0.6, 0.2),
                (0.3, 0.6),
                (0.4, 0.01),
                (50, -20)
                ]

    def test_calBasisFuncVal(self):
        expected = [[-4.176832772, 1.680848961],
                    [-4.476832772, 2.080848961],
                    [-4.376832772, 1.490848961],
                    [45.22316723, -18.51915104]]
        assert_array_almost_equal(expected,
                                  self.policy.calBasisFuncVal(self.feaList))

    def test_calSecondBasisFuncVal(self):
        expected = [[-196.7191887, 80.56815369],
                    [80.56815369, -33.04289481]]
        result = self.policy.calSecondBasisFuncVal(self.feaList)
        assert_array_almost_equal(expected, result)

    def test_getActionProb(self):
        last_ap = None
        for v in scipy.arange(-10, 10, 2):
            feaList = [
                    (v, 0.0),
                    (v, 0.3),
                    (v, 0.0),
                    (v, -0.3)
                    ]
            theta = scipy.array([10, 10]).reshape(-1, 1)
            ap = self.policy._getActionProb(feaList, theta)
            self.assertTrue(ap[1] > ap[3])
            self.assertEqual(ap[0], ap[2])
            last_ap = ap.tolist()


if __name__ == "__main__":
    unittest.main()


from __future__ import print_function, division, absolute_import
from . import util
import unittest
import scipy
from scipy.linalg import inv
from numpy.testing import assert_array_almost_equal, assert_almost_equal

class UtilTestCase(unittest.TestCase):

    def testEncodeTriuAs1DArray(self):
        triu = scipy.array([[1, 2, 3],
                            [2, 5, 6],
                            [3, 2, 9]])
        expected = [1, 2, 3, 5, 6, 9]
        assert_array_almost_equal(expected, util.encodeTriuAs1DArray(triu))

    def testDecode1DArrayAsTriu(self):
        expected = scipy.array([[1, 2, 3],
                                [0, 5, 6],
                                [0, 0, 9]])
        result = util.decode1DArrayAsTriu(scipy.array([1, 2, 3, 5, 6, 9]), 3)
        assert_array_almost_equal(expected, result)

    def testDecode1DArrayAsSymMat(self):
        expected = scipy.array([[1, 2, 3],
                                [2, 5, 6],
                                [3, 6, 9]])
        result = util.decode1DArrayAsSymMat(scipy.array([1, 2, 3, 5, 6, 9]), 3)
        assert_array_almost_equal(expected, result)


    def testShermanMorrisonUpdate(self):
        A = scipy.array([[1, 2],
                         [3, 4]])
        stepsize = 0.1
        z = scipy.array([4, 1])
        v = scipy.array([-2, 0])
        expected = inv(A + stepsize * (scipy.outer(z, v) - A))

        invA = inv(A)
        assert_array_almost_equal(expected, util.shermanMorrisonUpdate(invA,
                                                                       stepsize,
                                                                       z, v))

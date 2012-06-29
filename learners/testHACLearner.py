#!/usr/bin/env python
import unittest
from numpy import array, array_equal
from HACLearner import *
class HACLearnerTestCase(unittest.TestCase):
    def testOP(self):
        A = array([[1, 2],
                   [3, 4]])
        B = array([[2, 3],
                   [1, 2]])
        C = OP(A, B)
        C_should = array([[[2, 3], [6, 9]],
                         [[2, 4], [4, 8]]])
        self.assertTrue( np.array_equal(C, C_should) )
    def testROP(self):
        C = array([[[2, 0], [0, 4]],
                   [[5, 0], [0, 10]]])
        B = array([[1, 1],
                  [1, 1]])
        A = ROP(C, B)
        A_should = array([[0.5, 0.2],
                         [0.25, 0.1]])
        self.assertTrue( np.array_equal(A, A_should) )

if __name__ == "__main__":
    unittest.main()

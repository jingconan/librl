#!/usr/bin/env python
# from util import *
import sys
sys.path.append("..")
from util import GenRand, Expect
from numpy import array, exp, zeros, arange

class BoltzmanPolicy(object):
    """Policy is one type of controller that maps the observation
    to action probability
    PolicyController has two functions:
    1. Generate Actions Based on Observations and Weights
    2. Calculate basis function value. which is nabla_theta psi_theta(x,u)

    for bolzman distribution
            mu(u_i | x) = a_i(theta) / sum(a_i(theta))
            a_i(theta) = F_i(x) exp( theta_1 * E{safety( f(x,u_i) )} + theta_2 * E{progress( f(x,u_i) )} )

    """
    def __init__(self, T):
        self.T = T
        self.PU = None # PU is cached to accelarate the program
        self.g = None
        self.bf = None

    @staticmethod
    def CalW_EXP(score, theta, T):
        perfer = sum( s * t for s, t in zip(score, theta) )
        return float(exp(perfer/T))

    CalW = CalW_EXP

    def activate(self, feaList, theta):
        """ The output of activate function is a list containing the
        control sequence in the following n steps current parameter
        settings of policy. now n=1.
        """
        self.PU = self.getActionProb(feaList, theta)
        return [GenRand(self.PU)]

    def calBasisFuncVal(self, feaList):
        """for an observation, calculate value of basis function
        for all possible actions
            feaList is a list of tuple. each tuple represent the value of feature
            take the robot motion control as an example, a possible value may be:
                [
                ( safety_1 , progress_1),
                ( safety_2 , progress_2),
                ( safety_3 , progress_3),
                ( safety_4 , progress_4),
                ]
                1, 2, 3, 4 coressponds to each action ['E', 'N', 'W', 'S']
            ]
        """
        # FIXME be attention about usage of PU. use the cache value of PU
        fl = zip(*feaList)
        assert(self.PU)
        g = [Expect(flv, self.PU) for flv in fl]
        # FIXME ATTENTION, when taking derivative,  be careful about minus sign before etg.
        basisValue = array(feaList) - array(g).reshape(1, -1)
        self.g, self.bf = g, basisValue
        return basisValue


    def calSecondBasisFuncVal(self, feaList):
        assert( self.g is not None )
        assert( self.bf is not None )
        g, grad, PU = self.g, self.bf, self.PU

        uSize = len(feaList)
        n = len(feaList[0])
        assert(n == 2)

        es, ep = zip(*feaList)
        varsigma = zeros((uSize, n, n))
        for u in xrange(uSize):
            varsigma[u, 0, 0] = 1.0 / PU[u] * grad[u][0] * ( es[u] + float(g[1]) ) + \
                    sum( grad[j][0] * ( ep[j] ) for j in xrange(uSize))
            varsigma[u, 0, 1] = \
                    1.0 / PU[u] * grad[u][1] * ( es[u] + float(g[1]) ) + \
                    sum( grad[j][1] * ( ep[j]) for j in xrange(uSize))
            varsigma[u, 1, 0] = \
                    1.0 / PU[u] * grad[u][0] * ( ep[u] + float(g[0]) ) + \
                    sum( grad[j][0] * es[j] for j in xrange(uSize) )
            varsigma[u, 1, 1] = \
                    1.0 / PU[u] * grad[u][1] * ( ep[u] + float(g[0]) ) + \
                    sum( grad[j][1] * es[j] for j in xrange(uSize))

        self.PU = None; self.g = None; self.bf = None
        return varsigma

    def getActionProb(self, feaList, theta):
        PU = [ self.CalW([s, p], theta, self.T) for s, p in feaList ]
        s0 = sum(PU)
        PU = [p*1.0/s0 for p in PU]
        return PU

import unittest
class BoltzmanPolicyTestCase(unittest.TestCase):
    def setUp(self):
        print "In method", self._testMethodName
        self.policy = BoltzmanPolicy(100)

    def test_calBasisFuncValA(self):
        feaList = [
                (0.8114285714285716, 0.0),
                (0.8457142857142858, 0.6000000000000001),
                (0.8114285714285716, 0.0),
                (0.8457142857142858, -0.5999999999999996)
                ]
        self.policy.PU = [0.2, 0.2, 0.3, 0.3]

        basisValue = self.policy.calBasisFuncVal(feaList)
        # print 'basisValue, ', basisValue

    def test_calSecondBasisFuncVal(self):
        pass

    def test_getActionProb(self):
        last_ap = None
        for v in arange(-10, 10, 2):
            feaList = [
                    (v, 0.0),
                    (v, 0.3),
                    (v, 0.0),
                    (v, -0.3)
                    ]
            theta = array([10, 10]).reshape(-1, 1)
            ap = self.policy.getActionProb(feaList, theta)
            self.assertTrue(ap[1] > ap[3])
            self.assertEqual(ap[0], ap[2])
            self.assertNotEqual(ap, last_ap)
            last_ap = ap


if __name__ == "__main__":
    unittest.main()


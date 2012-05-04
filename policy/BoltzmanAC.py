from util import *
from numpy import array, exp, zeros


def CalW_EXP(score, theta, T):
    perfer = sum( s * t for s, t in zip(score, theta) )
    return exp(perfer/T)

class BoltzmanPolicy(object):
    """Policy is one type of controller that maps the observation
    to action probability
    PolicyController has two functions:
    1. Generate Actions Based on Observations and Weights
    2. Calculate basis function value. which is nabla_theta psi_theta(x,u)
    """
    def __init__(self, T):
        # self.CalW = CalW_EXP
        self.T = T
        self.PU = None # PU is cached to accelarate the program
        self.g = None
        self.bf = None

    def CalW(self, score, theta):
        return CalW_EXP(score, theta, self.T)

    def activate(self, feaList, theta):
        self.PU = self.getActionProb(feaList, theta)
        return [GenRand(self.PU)]

    def calBasisFuncVal(self, feaList):
        """for an observation, calculate value of basis function
        for all possible actions"""
        # FIXME be attention about usage of PU. use the cache value of PU
        fl = zip(*feaList)
        # assert(self.PU)
        g = [Expect(flv, self.PU) for flv in fl]
        # self.PU = None
        # FIXME ATTENTION, when taking derivative,  be careful about minus sign before etg.
        basisValue = array(feaList) - array(g).reshape(1, -1)
        # basisValue[:, 1] = -1 * basisValue[:, 1]
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
        PU = [ self.CalW([s, p], theta) for s, p in feaList ]
        s0 = sum(PU)
        PU = [p*1.0/s0 for p in PU]
        return PU














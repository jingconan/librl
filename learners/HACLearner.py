#!/usr/bin/env python
from LSTDACLearner import *
from copy import deepcopy
from pybrain.datasets.dataset import DataSet
from scipy import zeros, array, dot, log, diag
from scipy.linalg import pinv2 as pinv
from scipy.linalg import norm
from util import *

def OP(A, B):
    """& operatator defined"""
    assert(A.shape[0] == A.shape[1] == B.shape[0] == B.shape[1])
    n = A.shape[0];
    C = zeros([n, n, n])
    for i in xrange(n):
        C[i,:,:] = dot(A[:, i].reshape(-1, 1), B[i, :].reshape(1, -1))
    return C

def ROP(C, B):
    assert(C.shape[0] == C.shape[1] == C.shape[2] == B.shape[0] == B.shape[1])
    n = C.shape[0]
    A = zeros([n, n])
    for i in xrange(n):
        A[:, i] = dot( pinv(C[i,:,:]), B[:, i].reshape(-1, 1) ).reshape(-1)
    return A

class HACLearner(LSTDACLearner):
    def newEpisode(self):
        super(HACLearner, self).newEpisode()
        n = self.feadim
        self.Y = zeros([n, n])
        self.F = zeros([n, n])
        self.E = zeros([n, n, n])
        self.V = zeros([n, n])

        self.S = zeros([n, n])
        self.T = zeros([n, n])

        self.hessian = zeros([n, n])
        self.last_hDiff = None

    def Critic(self, gk, xkPsi, xkVarsigma, gkp1, xkp1Psi, xkp1Varsigma):
        super(HACLearner, self).Critic(gk, xkPsi, gkp1, xkp1Psi)
        # return
        n = self.feadim

        gam = 1.0 / (self.k+1)
        self.Y = self.lamb * self.Y + xkVarsigma
        self.F += gam * (gk * self.Y - self.F)
        delta = (xkp1Varsigma - xkVarsigma).T
        self.E += gam * ( OP(self.Y, delta) - self.E )
        self.S = -1 * ROP(self.E, self.F)
        # print 'self.F', self.F
        # print 'self.E', self.E

        h = dot(self.r.T, xkPsi) * xkPsi
        self.V += gam * ( dot(self.z.reshape(-1, 1), h.reshape(1, -1)) - self.V )
        self.T = -1 * dot( pinv(self.AE), self.V )

    def Actor(self, xkp1Psi, xkp1Varsigma):
        n = self.feadim; k = self.k; c = self.c; D = self.D

        # normR = norm(self.r)
        # tao = ( D / (normR + 0.0) ) if (normR > D) else 1
        # beta = 0 if (k == 0) else (c + 0.0 ) / ( (k+1) * log(k+1) )
        normR = norm(self.r)
        tao = ( D / (normR + 0.0) ) if (normR > D) else 1
        beta = 1 if (k <= 1) else ( (c + 0.0 ) / ( k * log(k) ) )

        gradLambda = dot(self.r.T, xkp1Psi) * xkp1Psi
        # H = dot( xkp1Varsigma, diag(diag(dot(self.S.T, xkp1Varsigma))) ) + \
                # dot( xkp1Psi, dot(xkp1Psi.T, self.T) )

        # self.theta = self.theta + beta * tao * gradLambda
        # return

        H = dot( xkp1Varsigma, diag(diag(dot(self.S, xkp1Varsigma))) ) + \
                dot( xkp1Psi, dot(xkp1Psi.T, self.T) )
        # self.hessian = 0.8 * self.hessian + H
        self.hessian = 0.1 * self.hessian + H
        # print 'self.hessian, ', self.hessian

        # hDiff = dot( pinv(H), gradLambda )
        # print 'xkp1Varsigma, ', xkp1Varsigma
        # print 'xkp1Psi', xkp1Psi
        # print 'self.hessian, ', self.hessian
        # print 'H, ', H
        # print 'self.S, ', self.S
        # import pdb;pdb.set_trace()
        hDiff = dot( pinv(self.hessian), gradLambda )
        assert( hDiff.shape[1] == 1)
        # print 'hDiff, ', hDiff.T.tolist()
        # print 'hDiff[0], ', hDiff.T.tolist()[0]
        # print 'gradLambda, ', gradLambda.T.tolist()[0]
        # print 'angle, ', angle(hDiff.T.tolist()[0], gradLambda.T.tolist()[0])
        ng = norm( gradLambda )
        nh = norm( hDiff )
        if nh < 10 and nh > 0:
            if self.last_hDiff is None:
                self.last_hDiff = hDiff
            # print 'self.hDiff', hDiff
            if angle(self.last_hDiff, hDiff) > 1:
                return
            self.theta = self.theta + beta * tao * hDiff / nh
        else:
            self.theta = self.theta + beta * tao * gradLambda

    def _updateWeights(self, xk, uk, gk, xkp1, ukp1, gkp1):
        xkPsi = self.module.calBasisFuncVal( self.to_list(xk) )
        xkVarsigma = self.module.calSecondBasisFuncVal( self.to_list(xk) )
        xkp1Psi = self.module.calBasisFuncVal( self.to_list(xkp1) )
        xkp1Varsigma = self.module.calSecondBasisFuncVal( self.to_list(xkp1) )
        # xkVarsigma = array([None, None, None, None])
        # xkp1Varsigma = array([None, None, None, None])
        self.Critic(gk, xkPsi[uk].reshape(-1, 1), xkVarsigma[uk],
                gkp1, xkp1Psi[ukp1].reshape(-1, 1), xkp1Varsigma[ukp1])
        self.Actor(xkp1Psi[ukp1].reshape(-1, 1), xkp1Varsigma[ukp1])

        self.k += 1

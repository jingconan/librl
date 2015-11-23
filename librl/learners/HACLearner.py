#!/usr/bin/env python
from LSTDACLearner import *
from copy import deepcopy
from pybrain.datasets.dataset import DataSet
from scipy import zeros, array, dot, log, diag
from scipy.linalg import pinv2 as pinv
from scipy.linalg import norm
from scipy.linalg import inv, LinAlgError
from scipy.linalg import cholesky
import numpy as np
import scipy.linalg
from librl.util import *

def OP(A, B):
    """& operatator defined
    for two nxn matrices A and B. the result will be
        C[i, :, :] = A[:, i] * B[i, :]
    """
    assert(A.shape[0] == A.shape[1] == B.shape[0] == B.shape[1])
    n = A.shape[0];
    C = zeros([n, n, n])
    for i in xrange(n):
        C[i,:,:] = dot(A[:, i].reshape(-1, 1), B[i, :].reshape(1, -1))
    return C

def ROP(C, B):
    """
        A[:, i] = C[i,:,:]^-1 * B[:, i]
    """
    assert(C.shape[0] == C.shape[1] == C.shape[2] == B.shape[0] == B.shape[1])
    n = C.shape[0]
    A = zeros([n, n])
    for i in xrange(n):
        A[:, i] = dot( pinv(C[i,:,:]), B[:, i].reshape(-1, 1) ).reshape(-1)
    return A

class HACLearner(LSTDACLearner):
    # def __init__(self, **kwargs):
        # super(HACLearner, self).__init__(**kwargs)
        # self.reset()

    def reset(self):
    # def newEpisode(self):
        super(HACLearner, self).reset()
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
        """
        The Critic of HAC is a supplyment of LSTD-AC. In addition to estimation
        the r for Q, The critic of HAC also estimation **S** and **T** that helps
        to estimate the hessian  matrix.
        """
        super(HACLearner, self).Critic(gk, xkPsi, gkp1, xkp1Psi)
        n = self.feadim

        gam = 1.0 / (self.k+1)
        self.Y = self.lamb * self.Y + xkVarsigma
        self.F += gam * (gk * self.Y - self.F)
        delta = (xkp1Varsigma - xkVarsigma).T
        self.E += gam * ( OP(self.Y, delta) - self.E )
        self.S = -1 * ROP(self.E, self.F)

        h = dot(self.r.T, xkPsi) * xkPsi
        self.V += gam * ( dot(self.z.reshape(-1, 1), h.reshape(1, -1)) - self.V )
        self.T = -1 * dot( pinv(self.AE), self.V )

    def Actor(self, xkp1Psi, xkp1Varsigma):
        """
        the Actor of HAC use Newton's Method to update the weight estimation.
        """
        n = self.feadim; k = self.k; c = self.c; D = self.D

        normR = norm(self.r)
        tao = ( D / (normR + 0.0) ) if (normR > D) else 1
        beta = 1 if (k <= 1) else ( (c + 0.0 ) / ( k * log(k) ) )

        ##### First Order information #######
        gradLambda = dot(self.r.T, xkp1Psi) * xkp1Psi

        ##### Second Order information #######
        H = dot( xkp1Varsigma, diag(diag(dot(self.S, xkp1Varsigma))) ) + \
                dot( xkp1Psi, dot(xkp1Psi.T, self.T) )
        self.hessian = 0.99 * self.hessian + H
        # print 'self.hessian', self.hessian

        # test #
        # print 'H, ', H
        # print 'self.S, ', self.S
        # print 'self.T, ', self.T
        # print 'xkp1Varsigma, ', xkp1Varsigma
        # print 'xkp1Psi', xkp1Psi
        # self.theta = self.theta + beta * tao * gradLambda
        # return

        # hDiff = dot( pinv(H), gradLambda )
        # hDiff = dot( pinv(self.hessian), gradLambda )
        try:
            # cho = scipy.linalg.cho_factor(self.hessian)
            #########################################
            ###  ensure H is positive definite ######
            #########################################
            large_val = np.max(np.abs(H))
            if large_val:
                H = H / np.max(np.abs(H))
            cho = scipy.linalg.cho_factor(H)
            hDiff  = scipy.linalg.cho_solve(cho, gradLambda)
            max_val = np.max(np.abs(hDiff))
            # print 'max_val', max_val
            if max_val > 50:
                hDiff = hDiff / max_val
            # print 'hDiff, ', hDiff
            # if max(hDiff) > 1000 or min(hDiff) < 1000:
                # self.theta = self.theta + beta * tao * gradLambda
                # return
            # hDiff = dot( inv(self.hessian), gradLambda )
            self.theta = self.theta + beta * tao * hDiff
            print 'update using hessian'
        except LinAlgError as e:
            # print 'LinAlgError'
            self.theta = self.theta + beta * tao * gradLambda


        # print hDiff
        # assert( hDiff.shape[1] == 1)
        # self.theta = self.theta + beta * tao * hDiff
        return

        ng = norm( gradLambda )
        nh = norm( hDiff )
        if nh < 10 and nh > 0:
            if self.last_hDiff is None:
                self.last_hDiff = hDiff
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
        self.Critic(gk, xkPsi[uk].reshape(-1, 1), xkVarsigma[uk],
                gkp1, xkp1Psi[ukp1].reshape(-1, 1), xkp1Varsigma[ukp1])
        self.Actor(xkp1Psi[ukp1].reshape(-1, 1), xkp1Varsigma[ukp1])

        self.k += 1

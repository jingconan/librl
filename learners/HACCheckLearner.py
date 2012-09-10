#!/usr/bin/env python
from HACLearner import HACLearner
from scipy import zeros, dot, log, diag
from scipy.linalg import pinv2 as pinv
from scipy.linalg import norm
# from util import *
class HACCheckLearner(HACLearner):
    def Critic(self, gk, xkPsi, xkVarsigma, gkp1, xkp1Psi, xkp1Varsigma):
        """
        The Critic of HAC is a supplyment of LSTD-AC. In addition to estimation
        the r for Q, The critic of HAC also estimation **S** and **T** that helps
        to estimate the hessian  matrix.
        """
        super(HACCheckLearner, self).Critic(gk, xkPsi, xkVarsigma, gkp1, xkp1Psi, xkp1Varsigma)

    def Actor(self, xkp1Psi, xkp1Varsigma):
        """
        the Actor of HAC use Newton's Method to update the weight estimation.
        """
        ##### First Order information #######
        gradLambda = dot(self.r.T, xkp1Psi) * xkp1Psi

        ##### Second Order information #######
        H = dot( xkp1Varsigma, diag(diag(dot(self.S, xkp1Varsigma))) ) + \
                dot( xkp1Psi, dot(xkp1Psi.T, self.T) )

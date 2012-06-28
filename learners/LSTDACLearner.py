#!/usr/bin/env python
__author__ = 'Jing Conan Wang, Boston University, wangjing@bu.edu'

import sys
sys.path.append("..")
# from pybrain.rl.learners.directsearch.policygradient import PolicyGradientLearner
# from pybrain.rl.learners.directsearch.policygradient import *
# from pybrain.utilities import *
# from ActorCriticLearner import *
# from util import *
import numpy as np
# from scipy import ones, dot, ravel, zeros, array
from TDLearner import TDLearner
from scipy import array, zeros, dot
from scipy.linalg import pinv2 as pinv
# from scipy.linalg import pinv

# bRec = []
# gRec = []
# class LSTDACLearner(PolicyGradientLearner):
class LSTDACLearner(TDLearner):
    ''' LSTD-Actor Critic Method. See `Least Squares Temporal Difference Actor-Critic Methods with Applications to Robot Motion Control <http://arxiv.org/abs/1108.4698>`_ for more information.
    Learner will learn from a data set. When the experiment is continous.
    the data set will only contains one point, and there will be some
    eligibility trace. Now we only consider the episodic case.
    '''
    # def __init__(self, **kwargs):
        # super(LSTDACLearner, self).__init__(**kwargs)
        # AE is the estimation of the difference of basis function

    def _init(self, policy, dataset):
        super(LSTDACLearner, self)._init(policy, dataset)
        self.AE = zeros( (self.feadim, self.feadim) )
        self.b = zeros( (self.feadim, 1) )

    def newEpisode(self):
        super(LSTDACLearner, self).newEpisode()
        n = self.feadim
        self.AE = zeros( (n, n) )
        self.b = zeros( (n, 1) )

    def Critic(self, gk, xkPsi, gkp1, xkp1Psi):
        # ----- Critic --------
        z, b, r, AE, lamb, alpha, k = self.z, self.b, self.r, self.AE, self.lamb, self.alpha, self.k
        xkPsi = xkPsi.reshape(-1, 1)
        xkp1Psi = xkp1Psi.reshape(-1, 1)
        z = lamb * z + xkPsi
        gam = 1.0 / (k+1)
        b += gam * ( z * gk - b )

        # Add by J.W
        # recFid = open('./record.txt', 'a')
        # recFid.write('%f %f %f\n'%(gk, b[0], b[1]))
        # recFid.close()

        psiTmp = xkp1Psi - xkPsi
        AE += gam * ( dot( z , (psiTmp.T ) ) - AE  )
        r = -1 * np.dot( pinv(AE), b )

        self.z, self.b, self.r, self.AE, self.alpha = z, b, r, AE, alpha


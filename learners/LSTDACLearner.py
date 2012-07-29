#!/usr/bin/env python
__author__ = 'Jing Conan Wang, Boston University, wangjing@bu.edu'

import sys
sys.path.append("..")
# from pybrain.rl.learners.directsearch.policygradient import PolicyGradientLearner
# from pybrain.rl.learners.directsearch.policygradient import *
# from pybrain.utilities import *
# from ActorCriticLearner import *
# from util import *
# from scipy import ones, dot, ravel, zeros, array
from TDLearner import TDLearner
from scipy import array, zeros, dot
from scipy.linalg import pinv2 as pinv

class LSTDACLearner(TDLearner):
    ''' LSTD-Actor Critic Method. See `Least Squares Temporal Difference Actor-Critic Methods with Applications to Robot Motion Control <http://arxiv.org/abs/1108.4698>`_ for more information.
    Learner will learn from a data set. When the experiment is continous.
    the data set will only contains one point, and there will be some
    eligibility trace. Now we only consider the episodic case.
    '''
    def _init(self, policy, dataset):
        super(LSTDACLearner, self)._init(policy, dataset)
        self.AE = zeros( (self.feadim, self.feadim) )
        self.b = zeros( (self.feadim, 1) )

    # def newEpisode(self):
    def reset(self):
        super(LSTDACLearner, self).newEpisode()
        n = self.feadim
        self.AE = zeros( (n, n) )
        self.b = zeros( (n, 1) )

    def Critic(self, gk, xkPsi, gkp1, xkp1Psi):
        # ----- Critic --------
        k = self.k;
        gam = 1.0 / (k+1)

        # self.alpha += gam * ( gk - self.alpha )
        self.z = self.lamb * self.z + xkPsi
        self.b += gam * ( (gk - self.alpha) * self.z  - self.b )
        psiTmp = xkp1Psi - xkPsi
        self.AE = self.AE + gam * ( dot( self.z , psiTmp.T ) - self.AE  )

        self.r = -1 * dot( pinv(self.AE), self.b )

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
from LSTDACLearner import LSTDACLearner
from scipy import array, zeros, dot
from scipy.linalg import pinv2 as pinv
# from scipy.linalg import pinv
from scipy.linalg import norm
from scipy import log

class NAC(LSTDACLearner):
    def Actor(self, xkp1Psi):
        # ------ Actor -------
        r, D, theta, c, k = self.r, self.D, self.theta, self.c, self.k

        normR = norm(r)
        tao = ( D / (normR + 0.0) ) if (normR > D) else 1
        beta = 1 if (k <= 1) else ( (c + 0.0 ) / ( k * log(k) ) )

        self.theta = self.theta + beta * tao * r

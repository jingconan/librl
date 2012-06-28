#!/usr/bin/env python
__author__ = 'Jing Conan Wang, Boston University, wangjing@bu.edu'

import sys
sys.path.append("..")
# from pybrain.rl.learners.directsearch.policygradient import PolicyGradientLearner
from pybrain.rl.learners.directsearch.policygradient import LoglhDataSet
# from pybrain.rl.learners.directsearch.policygradient import *
# from pybrain.utilities import *
from ActorCriticLearner import ActorCriticLearner
# from util import *
from scipy import dot, ravel, zeros, array, log
from scipy.linalg import norm

class TDLearner(ActorCriticLearner):
    """User TD Learner to learn the projection coefficient r of Q on the basis surface"""
    def __init__(self, actiondim, iniTheta, **kwargs):
        # PolicyGradientLearner.__init__(self)
        ActorCriticLearner.__init__(self)
        self.feadim = len(iniTheta)
        self.loglh = LoglhDataSet(self.feadim)
        self.theta = array(iniTheta).reshape(-1, 1)

        self.k = 0
        self.z = zeros( (self.feadim, 1) )
        self.r = zeros( (self.feadim, 1) )
        self.alpha = 0;
        self.lastobs = None

        # parameter
        self.lamb = kwargs['lamb']
        self.c = kwargs['c']
        self.D = kwargs['D']

    def newEpisode(self):
        self.k = 0
        n = self.feadim
        self.z = zeros( (n, 1) )
        self.r = zeros( (n, 1) )
        self.alpha = 0
        self.lastobs = None

    def setReachProbCal(self, reachProbCal):
        self.reachProbCal = reachProbCal

    def Critic(self, xk, uk, gk, xkPsi, xkp1, ukp1, gkp1, xkp1Psi):
        # ----- Critic --------
        gam = 1.0 / (self.k+1)
        d = gk - self.alpha + dot(self.r.T, xkp1Psi - xkPsi)
        self.alpha += gam * ( gkp1 - self.alpha )
        self.r += gam * d * self.z

        self.z = self.lamb * self.z + xkp1Psi

    def Actor(self, xkp1, ukp1, xkp1Psi):
        # ------ Actor -------
        r, D, theta, c, k = self.r, self.D, self.theta, self.c, self.k

        normR = norm(r)
        tao = ( D / (normR + 0.0) ) if (normR > D) else 1
        beta = 1 if (k <= 1) else ( (c + 0.0 ) / ( k * log(k) ) )

        theta = theta - beta * tao * dot(r.T, xkp1Psi) * xkp1Psi
        # theta = theta + beta * tao * np.dot(r.T, xkp1Psi) * xkp1Psi
        self.theta = theta

    def _updateWeights(self, xk, uk, gk, xkPsi, xkp1, ukp1, gkp1, xkp1Psi):
        self.Critic(xk, uk, gk, xkPsi, xkp1, ukp1, gkp1, xkp1Psi)
        self.Actor(xkp1, ukp1, xkp1Psi)
        self.k += 1

    def learnOnDataSet(self, dataset):
        self.dataset = dataset
        for n in range(self.dataset.getLength()):
            obs, action, reward = self.dataset.getLinked(n)
            seqidx = ravel(self.dataset['sequence_index'])
            # print 'self.dataset.getLength', self.dataset.getLength()
            if n == self.dataset.getLength() - 1:
                loglh = self.loglh['loglh'][seqidx[n], :]
            else:
                loglh = self.loglh['loglh'][seqidx[n]:seqidx[n + 1], :]

            if self.lastobs is not None:
                self._updateWeights(self.lastobs, self.lastaction[0], self.lastreward, self.lastloglh.reshape(-1, 1),
                        obs, action[0], reward, loglh.reshape(-1, 1))

            self.lastobs = obs
            self.lastaction = action
            self.lastreward = reward
            self.lastloglh = loglh

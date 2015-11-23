#!/usr/bin/env python
__author__ = 'Jing Conan Wang, Boston University, wangjing@bu.edu'

import sys
sys.path.append("..")
from pybrain.rl.learners.directsearch.policygradient import LoglhDataSet
from ActorCriticLearner import ActorCriticLearner
from scipy import dot, ravel, zeros, array, log
from scipy.linalg import norm

class TDLearner(ActorCriticLearner):
    """User TD Learner to learn the projection coefficient r of Q on the basis surface"""
    def __init__(self, **kwargs):
        ActorCriticLearner.__init__(self)
        self.module = None
        # parameter
        self.lamb = kwargs['lamb']
        self.c = kwargs['c']
        self.D = kwargs['D']

    def _init(self, policy, dataset):
        self.module = policy
        self.dataset = dataset
        self.feadim = len(self.theta)
        self.reset()
        self.newEpisode()

    def get_theta(self): return self.module.theta.reshape(-1, 1)
    def set_theta(self, val): self.module._setParameters(val.reshape(-1))
    theta = property(fget = get_theta, fset = set_theta)

    def resetStepSize(self):
        self.k = 0

    def reset(self):
        """reset all parameters"""
        self.k = 0
        n = self.feadim
        self.z = zeros( (n, 1) )
        self.r = zeros( (n, 1) )
        self.alpha = 0
        self.lastobs = None

    def newEpisode(self):
        """new Episode only restart the counter,
        not the parameter that has been estimated"""
        self.k = 0
        self.lastobs = None

    def setReachProbCal(self, reachProbCal):
        self.reachProbCal = reachProbCal

    def Critic(self, gk, xkPsi, gkp1, xkp1Psi):
        # ----- Critic --------
        gam = 1.0 / (self.k+1)
        d = gk - self.alpha + dot(self.r.T, xkp1Psi - xkPsi)
        self.alpha += gam * ( gkp1 - self.alpha )
        self.r += gam * d * self.z

        self.z = self.lamb * self.z + xkp1Psi

    def Actor(self, xkp1Psi):
        # ------ Actor -------
        r, D, theta, c, k = self.r, self.D, self.theta, self.c, self.k

        normR = norm(r)
        tao = ( D / (normR + 0.0) ) if (normR > D) else 1
        beta = 1 if (k <= 1) else ( (c + 0.0 ) / ( k * log(k) ) )

        # theta = theta - beta * tao * dot(r.T, xkp1Psi) * xkp1Psi
        theta = theta + beta * tao * dot(r.T, xkp1Psi) * xkp1Psi
        self.theta = theta

    def to_list(self, x):
        """change feature to list"""
        return x.reshape(-1, self.feadim).tolist()

    def _updateWeights(self, xk, uk, gk, xkp1, ukp1, gkp1):
        """Update weights of Critic and Actor based on the (state, action, reward) pair for
        current time and last time"""
        xkPsi = self.module.calBasisFuncVal( self.to_list(xk) )
        xkp1Psi = self.module.calBasisFuncVal( self.to_list(xkp1) )
        self.Critic(gk, xkPsi[uk].reshape(-1, 1), gkp1, xkp1Psi[ukp1].reshape(-1, 1))
        self.Actor(xkp1Psi[ukp1].reshape(-1, 1))
        self.k += 1

    def learnOnDataSet(self, dataset):
        """dataset is a sequence of (state, action, reward). update weights based on
        dataset"""
        self.dataset = dataset
        for n in range(self.dataset.getLength()):
            obs, action, reward = self.dataset.getLinked(n)
            if self.lastobs is not None:
                self._updateWeights(self.lastobs, self.lastaction[0], self.lastreward,
                        obs, action[0], reward)

            self.lastobs = obs
            self.lastaction = action
            self.lastreward = reward

#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

from scipy import dot, ravel, zeros, array, log, inner, outer, concatenate
from scipy.linalg import norm

from pybrain.rl.learners.directsearch.policygradient import LoglhDataSet
from pybrain.structure.modules.module import Module
from pybrain.structure.networks.network import Network
from pybrain.structure.parametercontainer import ParameterContainer

from .actorcritic import ActorCriticLearner
from ..policies.boltzmann import PolicyFeatureModule

class TDLearner(ActorCriticLearner):
    """User TD Learner to learn the projection coefficient r of Q on the basis surface"""
    def __init__(self, policy, **kwargs):
        ActorCriticLearner.__init__(self)
        self.module = None
        # parameter
        self.tracestepsize = kwargs['tracestepsize']
        self.actorstepsize = kwargs['actorstepsize']
        self.maxcriticnorm = kwargs['maxcriticnorm']

        self.module = PolicyFeatureModule(policy, 'policywrapper')
        self.feadim = len(self.module.theta)
        self.reset()
        self.newEpisode()

    def resetStepSize(self):
        self.k = 0

    def reset(self):
        """reset all parameters"""
        self.k = 0
        self.z = zeros((self.module.outdim,))
        self.r = zeros((self.module.outdim,))
        self.alpha = 0
        self.lastobs = None

    def newEpisode(self):
        """new Episode only restart the counter,
        not the parameter that has been estimated"""
        self.k = 0
        self.lastobs = None

    def setReachProbCal(self, reachProbCal):
        self.reachProbCal = reachProbCal

    def critic(self, lastreward, lastfeature, reward, feature):
        gam = 1.0 / (self.k+1)
        self.d = lastreward - self.alpha + inner(self.r, feature - lastfeature)
        self.alpha += gam * (reward - self.alpha)
        self.r += gam * self.d * self.z

        self.z = self.tracestepsize * self.z + feature

    def actor(self, obs, action, feature):
        normR = norm(self.r)
        tao = 1
        if normR > self.maxcriticnorm:
            tao = self.maxcriticnorm / (normR + 0.0)

        beta = 1
        if self.k > 1:
            beta = (self.actorstepsize + 0.0 ) / ( self.k * log(self.k) )


        self.module.theta += beta * tao * inner(self.r, feature) * \
                             feature[:self.feadim]

    def _updateWeights(self, lastobs, lastaction, lastreward, obs, action,
                       reward):
        """Update weights of Critic and Actor based on the (state, action, reward) pair for
        current time and last time"""
        lastfeature = self.module.activate(concatenate((lastobs, lastaction)))
        feature = self.module.activate(concatenate((obs, action)))
        self.critic(lastreward, lastfeature, reward, feature)
        self.actor(obs, action, feature)

    def learnOnDataSet(self, dataset):
        """dataset is a sequence of (state, action, reward). update weights based on
        dataset"""
        self.dataset = dataset
        for n in range(self.dataset.getLength()):
            obs, action, reward = self.dataset.getLinked(n)
            if self.lastobs is not None:
                self._updateWeights(self.lastobs, self.lastaction, self.lastreward,
                        obs, action, reward)
            self.k += 1

            self.lastobs = obs
            self.lastaction = action
            self.lastreward = reward

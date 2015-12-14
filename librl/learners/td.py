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
    def __init__(self, policy, tracestepsize, actorstepsize, maxcriticnorm,
                 module=None):
        ActorCriticLearner.__init__(self)
        # parameter
        self.tracestepsize = tracestepsize
        self.actorstepsize = actorstepsize
        self.maxcriticnorm = maxcriticnorm

        if module is None:
            self.module = PolicyFeatureModule(policy, 'policywrapper')
        else:
            self.module = module
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
        # Update critic parameter
        self.d = lastreward - self.alpha + inner(self.r, feature - lastfeature)
        self.r += self.gamma * self.d * self.z
        # Estimate of avg reward.
        self.alpha += self.gamma * (reward - self.alpha)
        # Update eligiblity trace
        self.z = self.tracestepsize * self.z + feature

    @property
    def gamma(self):
        return 1.0 / (self.k + 1)

    @property
    def beta(self):
        if self.k > 1:
            return (self.actorstepsize + 0.0 ) / ( self.k * log(self.k) )
        else:
            return 1

    def tao(self, r):
        normR = norm(r)
        if normR > self.maxcriticnorm:
            return self.maxcriticnorm / (normR + 0.0)
        else:
            return 1

    def stateActionValue(self, feature):
        return self.tao(self.r) * inner(self.r, feature)

    def actor(self, obs, action, feature):
        self.scaledFeature = (self.stateActionValue(feature) *
                              feature[:self.feadim])
        # Update policy parameter.
        # TODO(jingconanwang) somehow we cannot use += operator. Check the
        # reason.
        self.module.theta =  self.module.theta + self.beta * self.scaledFeature

    def _updateWeights(self, lastobs, lastaction, lastreward, obs, action,
                       reward):
        """Update weights of Critic and Actor based on the (state, action, reward) pair for
        current time and last time"""
        lastfeature = self.module.activate(concatenate((lastobs, lastaction)))
        feature = self.module.activate(concatenate((obs, action)))
        self.critic(lastreward, lastfeature, reward, feature)
        self.actor(obs, action, feature)

    def learnOnDataSet(self, dataset, startIndex=0, endIndex=None):
        """dataset is a sequence of (state, action, reward). update weights based on
        dataset"""
        self.dataset = dataset
        if endIndex is None:
            endIndex = dataset.getLength()
        assert endIndex <= dataset.getLength(), ('end index is larger '
                                                 'than dataset length')
        for n in range(startIndex, endIndex):
            obs, action, reward = self.dataset.getLinked(n)
            if self.lastobs is not None:
                self._updateWeights(self.lastobs, self.lastaction, self.lastreward,
                        obs, action, reward)
            self.k += 1

            self.lastobs = obs
            self.lastaction = action
            self.lastreward = reward

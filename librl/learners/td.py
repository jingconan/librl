#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import scipy
from scipy.linalg import norm
from .actorcritic import ActorCriticLearner

class TDLearner(ActorCriticLearner):
    """User TD Learner to learn the projection coefficient r of Q on the basis surface"""
    def __init__(self, policy, tracestepsize, actorstepsize, maxcriticnorm,
                 module=None):
        ActorCriticLearner.__init__(self, policy, module)
        # parameter
        self.tracestepsize = tracestepsize
        self.actorstepsize = actorstepsize
        self.maxcriticnorm = maxcriticnorm

    def reset(self):
        """reset all parameters"""
        self.resetStepSize()
        self.z = scipy.zeros((self.module.outdim,))
        self.r = scipy.zeros((self.module.outdim,))
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
        self.d = lastreward - self.alpha + scipy.inner(self.r, feature - lastfeature)
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
            return (self.actorstepsize + 0.0 ) / ( self.k * scipy.log(self.k) )
        else:
            return 1

    def tao(self, r):
        normR = norm(r)
        if normR > self.maxcriticnorm:
            return self.maxcriticnorm / (normR + 0.0)
        else:
            return 1

    def stateActionValue(self, feature):
        return self.tao(self.r) * scipy.inner(self.r, feature)

    def actor(self, obs, action, feature):
        self.scaledfeature = (self.stateActionValue(feature) *
                              feature[:self.paramdim])
        # Update policy parameter.
        # TODO(jingconanwang) somehow we cannot use += operator. Check the
        # reason.
        self.module.theta =  self.module.theta + self.beta * self.scaledfeature

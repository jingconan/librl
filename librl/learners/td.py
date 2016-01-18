#!/usr/bin/env python
from __future__ import print_function, division, absolute_import
import scipy
from scipy.linalg import norm
from .actorcritic import ActorCriticLearner

class TDLearner(ActorCriticLearner):
    """User TD Learner to learn the projection coefficient r of Q on the basis surface"""
    def __init__(self,
                 module,
                 cssinitial, cssdecay, # css means critic step size
                 assinitial, assdecay, # ass means actor steps size
                 rdecay, # reward decay weight
                 maxcriticnorm, # maximum critic norm
                 tracestepsize, # stepsize of trace
                 parambound = None # bound for the parameters
                 ):
        super(TDLearner, self).__init__(module)

        self.cssinitial = cssinitial
        self.cssdecay = cssdecay
        self.assinitial = assinitial
        self.assdecay = assdecay
        self.rdecay = rdecay
        self.maxcriticnorm = maxcriticnorm
        self.tracestepsize = tracestepsize

        if parambound is None:
            self.parambound = None
        else:
            self.parambound = scipy.array(parambound)

    def reset(self):
        """reset all parameters"""
        self.resetStepSize()
        self.z = scipy.zeros((self.criticdim,))
        self.r = scipy.zeros((self.criticdim,))
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
        if self.enableOnlyEssentialFeatureInCritic:
            lastfeature = self.module.decodeFeature(lastfeature, self.essentialFeature)
            feature = self.module.decodeFeature(feature, self.essentialFeature)

        # Estimate of avg reward.
        rweight = self.rdecay * self.gamma()
        self.alpha = (1 - rweight) * self.alpha + rweight * reward

        # Update critic parameter
        self.d = lastreward - self.alpha + scipy.inner(self.r, feature - lastfeature)
        self.r += self.gamma() * self.d * self.z

        # Update eligiblity trace
        self.z = self.tracestepsize * self.z + feature

    def gamma(self):
        return self.cssinitial * self.cssdecay / (self.cssdecay + self.k **
                                                  (2.0 / 3))

    # actor step size
    def beta(self):
        return self.assinitial * self.assdecay / (self.assdecay + self.k)

    def tao(self, r):
        normR = norm(r)
        if normR > self.maxcriticnorm:
            return self.maxcriticnorm / (normR + 0.0)
        else:
            return 1

    def stateActionValue(self, feature):
        r = self.tao(self.r) * self.r
        if self.enableOnlyEssentialFeatureInCritic:
            feature = self.module.decodeFeature(feature,
                                                self.essentialFeature)
        assert len(r) == self.criticdim, 'Wrong dimension of r'
        return scipy.inner(r, feature)

    def actor(self, lastobs, lastaction, lastfeature):
        safeature = self.module.decodeFeature(lastfeature, 'first_order')
        Q = self.stateActionValue(lastfeature)
        #  Q = self.d
        self.scaledfeature = Q * safeature
        # Update policy parameter.
        # TODO(jingconanwang) somehow we cannot use += operator. Check the
        # reason.
        self.module.theta =  self.module.theta + self.beta() * self.scaledfeature

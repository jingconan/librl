#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import scipy
from scipy import dot, ravel, zeros, array, log, inner, outer, concatenate
from scipy.linalg import norm, pinv, inv, LinAlgError
from .td import TDLearner
from ..util import shermanMorrisonUpdate

class LSTDLearner(TDLearner):
    actorUpdateInterval = 10
    criticResetInterval = 200
    def reset(self):
        super(LSTDLearner, self).reset()
        self.b = zeros((self.criticdim,))
        self.invA = scipy.eye(self.criticdim)
        self.A = scipy.eye(self.criticdim)
        self.alpha = 0

    def critic(self, lastreward, lastfeature, reward, feature):
        if self.k % self.criticResetInterval == 0:
            self.reset()

        # Estimate of avg reward.
        rweight = 1.0 / (self.k + 1)
        self.alpha = (1 - rweight) * self.alpha + rweight * reward
        gamma = self.gamma()
        fd = lastfeature - feature
        rd = lastreward - self.alpha

        # Update critic estimate
        self.b += rd * self.z
        self.A += outer(self.z, fd)

        # Update eligiblity trace
        self.z = self.tracestepsize * self.z + feature

    def updateCriticPara(self):
        # get inverse of A.
        try:
          self.invA = inv(self.A)
        except LinAlgError:
            pass

        self.r = dot(self.invA, self.b)

    def actor(self, lastobs, lastaction, lastfeature):
        if self.k % self.actorUpdateInterval != 0:
            return

        self.updateCriticPara()
        super(LSTDLearner, self).actor(lastobs, lastaction, lastfeature)

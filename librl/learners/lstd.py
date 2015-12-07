#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

from scipy import dot, ravel, zeros, array, log, inner, outer, concatenate
from scipy.linalg import norm, pinv
from .td import TDLearner

class LSTDLearner(TDLearner):
    def reset(self):
        super(LSTDLearner, self).reset()
        self.b = zeros((self.module.outdim,))
        self.A = zeros((self.module.outdim, self.module.outdim))

    def critic(self, lastreward, lastfeature, reward, feature):
        gam = 1.0 / (self.k+1)
        # Update critic parameter
        self.r = -1 * dot(pinv(self.A), self.b)
        # Update estimates
        self.b += (gam * (lastreward - self.alpha) * self.z - self.b)
        featureDifference = feature - lastfeature
        self.A += (gam * outer(self.z, featureDifference) - self.A)
        # Estimate of avg reward.
        self.alpha += gam * (reward - self.alpha)
        # Update eligiblity trace
        self.z = self.tracestepsize * self.z + feature

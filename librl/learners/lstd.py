#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import scipy
from scipy import dot, ravel, zeros, array, log, inner, outer, concatenate
from scipy.linalg import norm, pinv
from .td import TDLearner
from ..util import shermanMorrisonUpdate

class LSTDLearner(TDLearner):
    def reset(self):
        super(LSTDLearner, self).reset()
        self.b = zeros((self.criticdim,))
        self.invA = scipy.eye(self.criticdim)
        self.A = scipy.eye(self.criticdim)

    def critic(self, lastreward, lastfeature, reward, feature):
        self.invA = pinv(self.A)
        # Update critic parameter
        self.r = -1 * dot(self.invA, self.b)
        #  self.r = self.r * self.tao(self.r)
        # Update estimates
        self.b += self.gamma() * ((lastreward - self.alpha) * self.z -
                   self.b)
        featurediff = feature - lastfeature
        self.A += self.gamma() * (outer(self.z, featurediff) - self.A)
        # We use sherman-morrison inversion
        #  self.invA = shermanMorrisonUpdate(self.invA, self.gamma, self.z,
        #                                    featurediff)

        # Estimate of avg reward.
        self.alpha += self.gamma() * (reward - self.alpha)
        # Update eligiblity trace
        self.z = self.tracestepsize * self.z + feature

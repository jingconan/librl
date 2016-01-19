#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import scipy
from scipy import dot, ravel, zeros, array, log, inner, outer, concatenate
from scipy.linalg import norm, pinv, inv, LinAlgError
from .td import TDLearner
from ..util import shermanMorrisonUpdate

class LSTDLearner(TDLearner):
    def reset(self):
        super(LSTDLearner, self).reset()
        self.b = zeros((self.criticdim,))
        self.invA = scipy.eye(self.criticdim)
        self.A = scipy.eye(self.criticdim)

    def gamma(self):
        return 1.0 / (self.k + 1)

    def beta(self):
        return 1.0 / ((self.k + 2) * log(self.k + 2))

    def critic(self, lastreward, lastfeature, reward, feature):

        # Estimate of avg reward.
        rweight = self.rdecay * self.gamma()
        self.alpha = (1 - rweight) * self.alpha + rweight * reward
        gamma = self.gamma()
        fd = lastfeature - feature
        rd = lastreward - self.alpha

        #  self.r = self.r * self.tao(self.r)
        # Update critic estimate
        self.b = (1 - gamma) * self.b + gamma * rd * self.z
        self.A = (1 - gamma) * self.A + gamma * outer(self.z, fd)
        self.invA = pinv(self.A)
        if self.k % 10 == 0:
            self.r = dot(self.invA, self.b)

        #  self.d = rd - scipy.inner(self.r, fd)
        #  self.r += gamma * self.d * self.z

        # We use sherman-morrison inversion
        #  self.invA = shermanMorrisonUpdate(self.invA, self.gamma, self.z,
        #                                    featurediff)

        # Update eligiblity trace
        self.z = self.tracestepsize * self.z + feature

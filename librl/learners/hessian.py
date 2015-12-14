#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import scipy
from scipy import dot, zeros, inner, outer, concatenate, sqrt
from scipy.linalg import norm, pinv, inv, LinAlgError
from .lstd import LSTDLearner

class HessianLearner(LSTDLearner):

    def __init__(self, hessianlearningrate, *args, **kwargs):
        super(HessianLearner, self).__init__(*args, **kwargs)
        self.hessianlearningrate = hessianlearningrate
        self.hessiansamplenumber = 0

    def reset(self):
        super(HessianLearner, self).reset()
        self.eta = zeros((self.paramdim,))
        self.V = zeros((self.module.outdim, self.paramdim))
        self.T = zeros((self.module.outdim, self.paramdim))
        self.H = zeros((self.paramdim, self.paramdim))

    @property
    def gamma(self):
        return sqrt(self.zeta * self.beta)

    @property
    def zeta(self):
        return 1.0 / (self.k + 1)

    def stateActionValue(self, feature, weight=None):
        if weight is None:
            weight = self.r

        return self.tao(weight) * inner(feature, weight)

    def getHessianEstimate(self, feature):
        # Estimate gradient of state action value function w.r.t. parameters.
        n = self.paramdim
        # state-action value
        self.qvalue  = self.stateActionValue(feature, self.r)
        # gradient of the state-action value function w.r.t. the parameters
        qgradient = zeros((n,))
        for i in xrange(n):
            qgradient[i] = self.stateActionValue(feature, self.T[:, i])

        # The first n elements in the first-order basis (i.e., \nabla
        # \log(\mu)), the following n^2 elements are the second-order basis
        # (i.e., \nabla^2 \log(\mu)).
        loglhgrad = feature[0:n]
        loglhhessian = feature[n:(n+n*n)].reshape((n, n))

        term1 = self.qvalue * (loglhhessian - outer(loglhgrad, loglhgrad))
        term2 = outer(qgradient, loglhgrad)

        return term1 + term2 + term2.T

    def critic(self, lastreward, lastfeature, reward, feature):
        super(HessianLearner, self).critic(lastreward, lastfeature, reward,
                                           feature)

        # Update Q-tilde critic. Note that different with the notation in the
        # papar, we have minus here because of difference definition of
        # featureDifference.
        self.T = -1 * dot(self.invA, self.V)

        # Update estimate of Hessian
        self.U = self.getHessianEstimate(feature)
        try:
            self.H = self.hessianlearningrate * self.H + inv(self.U)
            self.hessiansamplenumber += 1
        except LinAlgError:
            pass

        # Update estimates
        self.scaledfeature = (self.stateActionValue(feature) *
                              feature[:self.paramdim])
        self.eta += self.zeta * self.scaledfeature
        vupdate = outer(self.z, self.scaledfeature - self.eta)
        self.V += self.zeta * (vupdate - self.V)

    def getScalingMatrix(self):
        # Here we add one to avoid division by zero.
        rho = 1.0 / (self.hessiansamplenumber + 1)
        return (1 - rho) *  self.H + rho * scipy.eye(self.paramdim)

    # It is intended that obs, action, and feature are not used.
    # scaledfeature has been calculated in critic and reused here.
    def actor(self, obs, action, feature):
        scaledgradient = dot(self.getScalingMatrix(), self.scaledfeature)
        self.module.theta =  self.module.theta + self.beta * scaledgradient

#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

import scipy
from scipy import dot, zeros, inner, outer, concatenate, sqrt
from scipy.linalg import norm, pinv, inv, LinAlgError
from .lstd import LSTDLearner
from .td import TDLearner

class HessianBase(object):
    rewardRange = [0, 400]
    def __init__(self, hessianlearningrate):
        self.hessianlearningrate = hessianlearningrate
        self.hessiansamplenumber = 0

    def gamma(self):
        return sqrt(self.zeta() * self.beta())

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
        self.loglhgrad = self.module.decodeFeature(feature, 'first_order')
        loglhhessian = self.module.decodeFeature(feature, 'second_order')

        term1 = self.qvalue * (loglhhessian - outer(self.loglhgrad, self.loglhgrad))
        term2 = outer(qgradient, self.loglhgrad)

        return term1 + term2 + term2.T

    def getScalingMatrix(self):
        # Here we add one to avoid division by zero.
        #  rho = 1.0 / (self.hessiansamplenumber + 1)
        if self.hessiansamplenumber < 30:
            rho = 0
        else:
            # get scale weight to reduce noise
            rr = self.rewardRange
            rho = (self.alpha - rr[0]) / (rr[1] - rr[0])
            rho = scipy.clip(rho, 0, 1)

        mat = rho *  self.H + (1 - rho) * scipy.eye(self.paramdim)
        return pinv(mat)

    # It is intended that obs, action, and feature are not used.
    # scaledfeature has been calculated in critic and reused here.
    def actor(self, obs, action, feature):
        scaledgradient = dot(self.getScalingMatrix(), self.scaledfeature)
        self.module.theta = self.ensureBound(self.module.theta + self.beta() *
                                             scaledgradient)

class HessianLearner(HessianBase, LSTDLearner):
    def __init__(self, hessianlearningrate, *args, **kwargs):
        HessianBase.__init__(self, hessianlearningrate)
        LSTDLearner.__init__(self, *args, **kwargs)

    def reset(self):
        LSTDLearner.reset(self)
        self.eta = zeros((self.paramdim,))
        self.V = zeros((self.module.outdim, self.paramdim))
        self.T = zeros((self.module.outdim, self.paramdim))
        self.H = zeros((self.paramdim, self.paramdim))

    #  def stateActionValue(self, *args, **kwargs):
    #      HessianBase.stateActionValue(self, *args, **kwargs)

    def critic(self, lastreward, lastfeature, reward, feature):
        LSTDLearner.critic(self, lastreward, lastfeature, reward,
                           feature)

        # Update Q-tilde critic. Note that different with the notation in the
        # papar, we have minus here because of difference definition of
        # featureDifference.
        self.T = dot(self.invA, self.V)

        # Update estimate of Hessian
        self.U = self.getHessianEstimate(feature)
        self.H = self.hessianlearningrate * self.H + self.U
        self.hessiansamplenumber += 1

        # Update estimates
        self.scaledfeature = self.stateActionValue(feature) * self.loglhgrad
        self.eta += self.zeta() * self.scaledfeature
        vupdate = outer(self.z, self.scaledfeature - self.eta)
        self.V += self.zeta() * (vupdate - self.V)


class HessianTDLearner(HessianBase, TDLearner):
    def __init__(self, hessianlearningrate, *args, **kwargs):
        HessianBase.__init__(self, hessianlearningrate)
        TDLearner.__init__(self, *args, **kwargs)

    def reset(self):
        TDLearner.reset(self)
        self.eta = zeros((self.paramdim,))
        self.T = zeros((self.module.outdim, self.paramdim))
        self.H = zeros((self.paramdim, self.paramdim))

    def critic(self, lastreward, lastfeature, reward, feature):
        TDLearner.critic(self, lastreward, lastfeature, reward,
                         feature)
        zeta = self.zeta()

        ff = self.module.decodeFeature(feature, 'first_order')
        preward = self.stateActionValue(feature) * ff
        lff = self.module.decodeFeature(lastfeature, 'first_order')
        lastpreward = self.stateActionValue(lastfeature) * lff
        self.scaledfeature = lastpreward

        # Estimate of "avg reward".
        rweight = self.rdecay * self.gamma()
        self.eta = (1 - rweight) * self.eta + rweight * preward

        fd = lastfeature - feature
        rd = lastpreward - self.eta

        self.D = rd - scipy.dot(fd.reshape((1, -1)), self.T).reshape(-1)
        self.T += zeta * scipy.outer(self.z, self.D)

        # Update estimate of Hessian
        self.U = self.getHessianEstimate(feature)
        self.H = self.hessianlearningrate * self.zeta() * self.H + self.U
        self.hessiansamplenumber += 1

#!/usr/bin/env python
from __future__ import print_function, division, absolute_import

from scipy import dot, ravel, zeros, array, log, inner, outer, concatenate
from scipy.linalg import norm

from pybrain.rl.learners.directsearch.policygradient import LoglhDataSet
from pybrain.structure.modules.module import Module
from pybrain.structure.networks.network import Network
from pybrain.structure.parametercontainer import ParameterContainer

from .actorcritic import ActorCriticLearner


class PolicyFeatureModule(Module):
    def __init__(self, policy, name=None):
        self.policy = policy
        self.feadim = policy.feadim
        self.actionnum = policy.actionnum
        self.featuredescriptor = self.getFeatureDescritor()

        outdim = 0
        for feature in self.featuredescriptor:
            outdim += feature['dimension']

        super(PolicyFeatureModule, self).__init__(
            indim = self.feadim * self.actionnum + 1,
            outdim = outdim,
            name = name
        )

    def getFeatureDescritor(self):
        def _firstOrderFeature(policy, feature, action):
            return self.getFeatureSlice(policy.calBasisFuncVal(feature).reshape(-1),
                                        action)

        def _secondOrderFeature(policy, feature, action):
            return policy.calSecondBasisFuncVal(feature).reshape(-1)

        return [
            {
                'dimension': self.feadim,
                'constructor': _firstOrderFeature,
            },
            {
                'dimension': self.feadim * self.feadim,
                'constructor': _secondOrderFeature,
            },
        ]

    def getFeatureSlice(self, feature, action):
        featureslice = self.policy.obs2fea(feature[0:(self.feadim *
                                                      self.actionnum)])
        return featureslice[action, :]

    def _forwardImplementation(self, inbuf, outbuf):
        fea = self.policy.obs2fea(inbuf[:-1])
        action = inbuf[-1]
        offset = 0
        for desc in self.featuredescriptor:
            newoffset = offset + desc['dimension']
            outbuf[offset:newoffset] = desc['constructor'](self.policy,
                                                           fea, action)
            offset = newoffset

    def get_theta(self): return self.policy.theta.reshape(-1)
    def set_theta(self, val): self.policy._setParameters(val.reshape(-1))
    theta = property(fget = get_theta, fset = set_theta)




class TDLearner(ActorCriticLearner):
    """User TD Learner to learn the projection coefficient r of Q on the basis surface"""
    def __init__(self, policy, dataset, **kwargs):
        ActorCriticLearner.__init__(self)
        self.module = None
        # parameter
        self.tracestepsize = kwargs['tracestepsize']
        self.actorstepsize = kwargs['actorstepsize']
        self.maxcriticnorm = kwargs['maxcriticnorm']

        self._init(policy, dataset)

    def _init(self, policy, dataset):
        self.module = PolicyFeatureModule(policy, 'policywrapper')
        self.dataset = dataset
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

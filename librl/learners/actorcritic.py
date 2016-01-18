from __future__ import print_function, division, absolute_import
import scipy
from ..policies.boltzmann import PolicyFeatureModule

class ActorCriticLearner(object):
    """This is the basis class for all actor-critic method"""
    def __init__(self, module,
                 enableOnlyEssentialFeatureInCritic=False,
                 essentialFeature='first_order'):
        self.module = module
        self.paramdim = len(self.module.theta)
        self.enableOnlyEssentialFeatureInCritic = enableOnlyEssentialFeatureInCritic
        if self.enableOnlyEssentialFeatureInCritic:
            self.criticdim = self.module.feadesc[self.essentialFeature]['dimension']
        else:
            self.criticdim = self.module.outdim

        self.reset()
        self.newEpisode()

    def critic(self, lastreward, lastfeature, reward, feature):
        abstractMethod()

    def actor(self, lastobs, lastaction, lastfeature):
        abstractMethod()

    def resetStepSize(self):
        self.k = 0

    def _updateWeights(self, lastobs, lastaction, lastreward, obs, action,
                       reward):
        """Update weights of Critic and Actor based on the (state, action, reward) pair for
        current time and last time"""
        lastfeature = self.module.activate(scipy.concatenate((lastobs, lastaction)))
        feature = self.module.activate(scipy.concatenate((obs, action)))
        self.critic(lastreward, lastfeature, reward, feature)
        self.actor(lastobs, lastaction, lastfeature)

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

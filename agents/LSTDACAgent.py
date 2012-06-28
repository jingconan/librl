__author__ = 'Jing Conan Wang, wangjing@bu.edu'

from pybrain.rl.agents.logging import LoggingAgent
from util import debug

class LSTDACAgent(LoggingAgent):
    '''Agent for LSTD AC algorithm

    .integrateObservation()
    .getAction()
    .giveReward()
    .learn()

    the usage of Agent is to extract the feature for each state.
    store them together and call learner.
    '''
    def __init__(self, policy, learner, sdim, adim):
        LoggingAgent.__init__(self, sdim, adim)
        self.policy = policy
        self.learner = learner
        self.lastaction = [0]
        self.feaList = None
        self.lastBasisFuncValue = None

        # if learner is available, tell it the module and data
        # if self.learner is not None:
            # self.learner.module = self.policy
            # self.learner.dataset = self.history

        # self.learning = True

    @debug
    def integrateObservation(self, obs):
        """Observation includes the robotion's place and also the feature list.
        e.g. obs = [(xPos, yPos), [(s1, p1), (s2, p2), ..., ]]"""
        # print 'obs2,', obs
        # LoggingAgent.integrateObservation(self, obs[0])
        # _, self.feaList = obs

        LoggingAgent.integrateObservation(self, obs[0])
        # self.feaList = obs.reshape()

    @debug
    def getAction(self):
        """This is basically the Actor part"""
        LoggingAgent.getAction(self)
        self.lastaction = self.policy.activate(self.feaList, self.learner.theta)
        self.lastBasisFuncValue = self.policy.calBasisFuncVal(self.feaList)
        return self.lastaction

    @debug
    def giveReward(self, r):
        LoggingAgent.giveReward(self, r)

        if self.logging:
            self.learner.loglh.appendLinked(self.lastBasisFuncValue[self.lastaction[0]])

    @debug
    def learn(self):
        self.learner.learnOnDataSet(self.history)
        self.history.clear()
        self.learner.loglh.clear()

    def newEpisode(self):
        self.history.clear()
        self.learner.loglh.clear()
        self.learner.newEpisode()
        if self.logging:
            self.history.newSequence()

    def reset(self):
        self.history.clear()
        self.learner.loglh.clear()
        self.learner.newEpisode()
        self.lastaction = [0]





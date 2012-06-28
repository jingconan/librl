__author__ = 'Jing Conan Wang, wangjing@bu.edu'

from pybrain.rl.agents.logging import LoggingAgent
from util import debug

class ACAgent(LoggingAgent):
    '''Agent for Actor Crictic algorithm

    .integrateObservation()
    .getAction()
    .giveReward()
    .learn()

    the usage of Agent is to extract the feature for each state.
    store them together and call learner.
    '''
    def __init__(self, policy, learner, sdim, adim=1):
        LoggingAgent.__init__(self, sdim, adim)
        self.policy = policy
        self.learner = learner
        self.lastaction = [0]
        self.feaList = None
        self.lastBasisFuncValue = None

        # if learner is available, tell it the module and data
        if self.learner is not None:
            self.learner._init(self.policy, self.history)

        self.learning = True

    @debug
    def getAction(self):
        """This is basically the Actor part"""
        LoggingAgent.getAction(self)
        self.lastaction = self.policy.activate(self.lastobs)
        return self.lastaction

    @debug
    def learn(self):
        self.learner.learnOnDataSet(self.history)
        self.history.clear()
        self.learner.loglh.clear()

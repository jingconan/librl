__author__ = 'Jing Conan Wang, wangjing@bu.edu'

from pybrain.rl.agents.logging import LoggingAgent

class ActorCriticAgent(LoggingAgent):
    """Agent for Actor Crictic algorithm

    .integrateObservation()
    .getAction()
    .giveReward()
    .learn()

    the usage of Agent is to extract the feature for each state.
    store them together and call learner.
    """

    def __init__(self, learner, sdim, adim=1, maxHistoryLength=1000):
        LoggingAgent.__init__(self, sdim, adim)
        self.learner = learner
        self.policy = self.learner.module.policy
        self.lastaction = None
        self.learning = True

        self.currentDataIndex = 0
        self.maxHistoryLength = maxHistoryLength

    def getAction(self):
        """This is basically the Actor part"""
        LoggingAgent.getAction(self)
        self.lastaction = self.policy.activate(self.lastobs)
        return self.lastaction

    #TODO(jingconanwang): extend learn to handle episodic experiment. Right
    #now we focus on average reward experiment.
    def learn(self):
        historyLength = self.history.getLength()
        if historyLength >= 2:
            self.learner.learnOnDataSet(self.history,
                                        self.currentDataIndex,
                                        self.currentDataIndex+1)
            self.currentDataIndex += 1

        if historyLength > self.maxHistoryLength:
            self.history.clear()
            self.currentDataIndex = 0

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
    def __init__(self, learner, sdim, adim=1):
        LoggingAgent.__init__(self, sdim, adim)
        self.learner = learner
        self.policy = self.learner.module.policy
        self.lastaction = None
        self.learning = True

    def getAction(self):
        """This is basically the Actor part"""
        LoggingAgent.getAction(self)
        self.lastaction = self.policy.activate(self.lastobs)
        return self.lastaction

    def learn(self):
        self.learner.learnOnDataSet(self.history)
        self.history.clear()

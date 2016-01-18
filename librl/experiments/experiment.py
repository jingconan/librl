import scipy
from pybrain.rl.experiments import Experiment
from librl.util import cPrint

class SessionExperiment(Experiment):
    def __init__(self, task, agent, policy):
        self.policy = policy
        super(SessionExperiment, self).__init__(task, agent)

    def doInteractionsAndLearn(self, number = 1000):
        reward = 0
        for j in xrange(number):
            reward += self._oneInteraction()
        self.agent.learn()
        return reward

    def doSessionsAndPrint(self, sessionNumber, sessionSize,
                           customPrinter=None):
        # periodically reset stepsize to increase learning speed.
        for i in xrange(sessionNumber):
            reward = self.doInteractionsAndLearn(sessionSize)

            if i % 1000 == 0:
                self.agent.learner.resetStepSize()
            if customPrinter:
                customPrinter()
            else:
                cPrint(iteration=i,
                       th_max=max(self.policy.theta),
                       th_min=min(self.policy.theta),
                       th_mean=scipy.mean(self.policy.theta),
                       #  th_std=scipy.std(policy.theta),
                       #  obs=sum(agent.lastobs),
                       reward=reward)

__author__ = 'Jing Conan Wang, wangjing@bu.edu'
import sys
sys.path.insert(0, "..")
from pybrain.rl.agents.learning import LearningAgent, LoggingAgent
from explorers.NULLExplorer import NULLExplorer


class ExplorerLearningAgent(LearningAgent):
    def __init__(self, module, learner):
        super(ExplorerLearningAgent, self).__init__(module, learner)
        # explorer.module = module
        learner.module = module
        # print 'module, theta', module.theta
        # import pdb;pdb.set_trace()
        # learner.network._setParameters(module.theta)
        # learner.explorer = explorer
        learner.explorer = NULLExplorer()

    def getAction(self):
        """ Activate the module with the last observation, add the exploration from
            the explorer object and store the result as last action. """
        LoggingAgent.getAction(self)
        self.lastaction = self.module.activate(self.lastobs)
        basis = self.module.calBasisFuncVal(self.module.obs2fea(self.lastobs))
        # self.learner.loglh.appendLinked( -1 * basis[self.lastaction[0]] )
        # print 'self, ', basis[self.lastaction[0]]
        self.learner.loglh.appendLinked( basis[self.lastaction[0]] )
        return self.lastaction

    def learn(self, episodes=1):
        if self.learning:
            self.learner.learnEpisodes(episodes)
        self.learner.loglh.clear()
        self.learner.dataset.clear()

__author__ = 'Jing Conan Wang, wangjing@bu.edu'
import sys
sys.path.insert(0, "..")
from pybrain.rl.agents.learning import LearningAgent


class ExplorerLearningAgent(LearningAgent):
    def __init__(self, module, learner, explorer):
        super(ExplorerLearningAgent, self).__init__(module, learner)
        explorer.module = module
        learner.module = module
        learner.explorer = explorer

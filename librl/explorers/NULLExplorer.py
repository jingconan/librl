from pybrain.rl.explorers.discrete import BoltzmannExplorer
class NULLExplorer(BoltzmannExplorer):
    def activate(self, state, action):
        return action

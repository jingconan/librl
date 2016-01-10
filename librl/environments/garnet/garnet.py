from __future__ import print_function, division, absolute_import
import scipy
from pybrain.rl.environments.environment import Environment
from pybrain.rl.environments.task import Task
from librl.util import zdump, zload

# FIXME(hbhzwj) add unittest for GarnetTask and GarnetEnvironment.
class GarnetTask(Task):
    """Garnet Task

    The expected reward for each transition is a normally distributed random
    variable with mean 0 and unit variance. The actual reward id selected
    randomly according to a normal distribution with mean equal to the
    expected reward and standard deviation \sigma.
    """
    def __init__(self, environment, sigma):
        super(GarnetTask, self).__init__(environment)
        self.sigma = sigma
        self.rewardCache = dict()
        self.numStates = self.env.numStates
        self.numActions = self.env.numActions

    def _getExpectedReward(self, key):
        expectedReward = self.rewardCache.get(key)
        if expectedReward is not None:
            return expectedReward
        self.rewardCache[key] = scipy.random.randn()
        return self.rewardCache[key]

    def performAction(self, action):
        self.env.lastAction = action
        super(GarnetTask, self).performAction(action)

    def getReward(self):
        key = (self.env.prevState, self.env.curState,
               int(self.env.lastAction[0]))
        expectedReward = self._getExpectedReward(key)
        reward = self.sigma * scipy.random.randn() + expectedReward
        return reward

    def getObservation(self):
        sensors = self.env.getSensors()
        fd = len(sensors)
        feature = scipy.zeros((self.numActions, fd * self.numActions))
        for i in xrange(self.numActions):
            feature[i, (i*fd):((i+1)*fd)] = sensors
        return feature.reshape(-1)

class GarnetEnvironment(Environment):
    """Generic Average Reward Non-stationary Environment TestBed

    Parameters
    -----
    numStates : int
        # of states.
    numActions : int
        # of actions
    branching : int
        # of possible next states for each state-action pair.
    feaDim, feaSum: int
        the feature for each state is a vector of {0, 1} whose length is feaDim
        and whose sum is feaSum.

    """
    initState = 0
    def __init__(self, numStates, numActions, branching, feaDim, feaSum,
                 savePath=None, loadPath=None):
        self.numStates = numStates
        self.numActions = numActions
        self.branching = branching
        self.feaDim = feaDim
        self.feaSum = feaSum

        if loadPath is not None:
            self._load(loadPath)
        else:
            self._genTransitionTable()
            self._genStateObs()

        if savePath is not None:
            self._save(savePath)

        self.curState = self.initState
        # null value for action
        self.lastAction = scipy.array([-1])

    def _save(self, savePath):
        message = dict()
        message['transitionStates'] = self.transitionStates
        message['transitionProb'] = self.transitionProb
        message['stateObs'] = self.stateObs
        zdump(message, savePath)

    def _load(self, loadPath):
        message = zload(loadPath)
        self.transitionStates = message['transitionStates']
        self.transitionProb = message['transitionProb']
        self.stateObs = message['stateObs']

    def _genStateObs(self):
        pos = range(self.feaDim)
        stateObs = set()
        while len(stateObs) < self.numStates:
            obs = scipy.zeros((self.feaDim,), dtype=int)
            ones = scipy.random.choice(pos, self.feaSum, False)
            obs[ones] = 1
            stateObs.add(tuple(obs.tolist()))
        self.stateObs = list(stateObs)

    def _genTransitionTable(self):
        # generate state that will be transited to and corresponding
        # probabilities.
        self.transitionStates = scipy.zeros((self.numActions, self.numStates, self.branching))
        self.transitionProb = scipy.zeros((self.numActions, self.numStates, self.branching))
        allStates = scipy.arange(self.numStates)
        for u in xrange(self.numActions):
            for i in xrange(self.numStates):
                self.transitionStates[u, i, :] = scipy.random.choice(allStates,
                                                                     size=self.branching,
                                                                     replace=False)
                # probabilities
                cutPoints = scipy.random.rand(self.branching-1)
                cutPoints = sorted(cutPoints.tolist() + [0, 1])
                self.transitionProb[u, i, :] = scipy.diff(cutPoints)

    def getSensors(self):
        return self.stateObs[self.curState]

    def performAction(self, action):
        action = action[0]
        states = self.transitionStates[action, self.curState, :]
        prob = self.transitionProb[action, self.curState, :]
        nextState = scipy.random.choice(states, size=1, p=prob)
        nextState = int(nextState[0])
        self.prevState = self.curState
        self.curState = nextState

    def reset(self):
        self.curState = self.initState

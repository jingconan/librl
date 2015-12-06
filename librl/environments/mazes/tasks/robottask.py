import collections
import scipy

from pybrain.rl.environments.mazes import MDPMazeTask

def cityBlockDistance(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])

class RobotMotionAvgRewardTask(MDPMazeTask):
    """This is a MDP Maze Task for Robot Motion Control.
    There are some goal states and some unsafe states.
    If the robot reach unsafe states, the reward(cost)
    will be 1. otherwise the reward will be zero. We want
    to maximize the average reward received by the robot.
    """
    TRAP_REWARD = -1
    GOAL_REWARD = 1
    DEFAULT_REWARD = 0

    TOLERANCE = 1e-7

    def __init__(self, environment, senseRange):
        MDPMazeTask.__init__(self, environment)
        self.senseRange = senseRange

        # find goal states from env mask.
        self.goalStates = []
        for i in xrange(self.env.mazeSize[0]):
            for j in xrange(self.env.mazeSize[1]):
                if self.env.isGoal((i, j)):
                    self.goalStates.append((i, j))

        self.cacheTranProbToNextStates = {}
        self.cacheMinDistanceToGoal = {}

    def reset(self):
        self.env.reset()

    def getObservation(self):
        """the agent receive its
        1. featureList for the state and for each possible action in the future.
           now featureList is [safety, progress]
        if there are four possible controls. then agen will receive 8x1 array.
        """
        return self.getFeature(self.env.perseus).reshape(-1)

    def performAction(self, action):
        super(RobotMotionTask, self).performAction(action)

    def getReward(self):
        ''' compute and return the current reward (i.e.
        corresponding to the last action performed) '''
        if self.env.isTrap(self.env.perseus):
            reward = self.TRAP_REWARD
        elif self.env.isGoal(self.env.perseus):
            reward = self.GOAL_REWARD
        else:
            reward = self.DEFAULT_REWARD

        return reward

    def getMinDistanceToGoal(self, state):
        """Get the minimium distance to any of goal stats."""
        searchKey = tuple(state)
        cacheValue = self.cacheMinDistanceToGoal.get(searchKey)
        if cacheValue: return cacheValue

        distance = float('inf')
        for goal in self.goalStates:
            tmp = cityBlockDistance(goal, state)
            if tmp < distance:
                distance = tmp

        self.cacheMinDistanceToGoal[searchKey] = distance
        return distance

    def getSafetyScore(self, state):
        return 1.0 - self.env.isTrap(state)

    def getFeature(self, state):
        """We can get features for each state. it may be
        distance to goal, safety degree. e.t.c"""
        # cache the feature result, boost the speed.
        numActions = self.env.numActions
        # now we only support two features
        feature = scipy.zeros((numActions, 2))
        for i in xrange(numActions):
            # create a action probability arry for only moving to one
            # direction.
            tmp = [0] * numActions
            tmp[i] = 1
            actionProb = tuple(tmp)

            tranProb = self.getMultiStepTranProb(state, self.senseRange,
                                                 actionProb)
            for s, p in tranProb.iteritems():
                # safety score
                feature[i, 0] += self.getSafetyScore(s) * p
                # progress score
                feature[i, 1] += -1.0 * self.getMinDistanceToGoal(s) * p

        return feature

    def getMultiStepTranProb(self, state, step, actionProb):
        """Get tranistion probability to allowable next step,
        allowns is the set of allowble next state
        allowTP is the transition probability under differnece action. it is a list of list,
        the 1st dimension = control size. the 2nd dimension = len(allowns)

        Args:
            state: tuple, a tuple of position.
            step: int, the # of steps
            actionProb: tuple of probablity. it should sum to 1.
        Return:
            a defaultdict whose key is the step and value is the transition
            probablity under actionProb.
        """
        assert abs(sum(actionProb) - 1) < self.TOLERANCE, ("action probability "
                                                           "doesn't sum to 1.")
        searchKey = tuple(state) + (step,) + tuple(actionProb)
        cacheValue = self.cacheTranProbToNextStates.get(searchKey)
        if cacheValue:
            return cacheValue

        result = collections.defaultdict(float)
        if step == 0:
            result[state] = 1
            return result

        for cIndex, control in enumerate(self.env.allActions):
            if abs(actionProb[cIndex] < self.TOLERANCE):
                continue

            sumProb= 0.0
            validActions = []
            for aIndex, actual in enumerate(self.env.allActions):
                nextState = (state[0] + actual[0], state[1] + actual[1])
                tranProbToNextPos = self.env.tranProb[cIndex][aIndex]
                # sum probablies of valid next states
                if not self.env.isOutBound(nextState):
                    sumProb += tranProbToNextPos
                    validActions.append((aIndex, actual, nextState, tranProbToNextPos))

            for aIndex, actual, nextState, tranProbToNextPos in validActions:
                recursiveStates = self.getMultiStepTranProb(nextState,
                                                            step - 1,
                                                            actionProb)
                sumRecursiveProb = 0
                for s, p in recursiveStates.iteritems():
                    sumRecursiveProb += p
                    # calculate the scale factor so that the final tranistion
                    # probability sums to 1 for each action.
                    result[s] += p * (tranProbToNextPos / sumProb) * actionProb[cIndex]

                errMsg = "transition probability doesn't sum to 1."
                assert abs(sumRecursiveProb - 1.0) <= 1e-3, errMsg

        self.cacheTranProbToNextStates[searchKey] = result
        return result

    def isFinished(self):
        return False

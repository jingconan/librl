import collections
import scipy

from pybrain.rl.environments.mazes import MDPMazeTask


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
        self.cacheTranProbToNextStates = {}

    def reset(self):
        self.env.reset()

    @property
    def outdim(self):
        return self.env.numActions * self.env.outdim

    def getObservation(self):
        """the agent receive its
        1. featureList for the state and for each possible action in the future.
           now featureList is [safety, progress]
        if there are four possible controls. then agen will receive 8x1 array.
        """
        return self.getFeature(self.env.perseus).reshape(-1)

    def performAction(self, action):
        super(RobotMotionAvgRewardTask, self).performAction(action)

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

    def getFeature(self, state):
        """We can get features for each state. it may be
        distance to goal, safety degree. e.t.c"""
        # cache the feature result, boost the speed.
        numActions = self.env.numActions
        # now we only support two features
        curSensors = self.env.getSensors(state)

        feature = scipy.zeros((numActions, self.env.outdim))
        for i in xrange(numActions):
            # create a action probability arry for only moving to one
            # direction.
            tmp = [0] * numActions
            tmp[i] = 1
            actionProb = tuple(tmp)

            tranProb = self.getMultiStepTranProb(state, self.senseRange,
                                                 actionProb)
            for s, p in tranProb.iteritems():
                feature[i, :] += (self.env.getSensors(s) * p)
            feature[i, :] -= curSensors

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

class RobotMotionAvgRewardWithStateObsTask(RobotMotionAvgRewardTask):
    def getObservation(self):
        """the agent receive its
        1. featureList for the state and for each possible action in the future.
           now featureList is [safety, progress]
        if there are four possible controls. then agen will receive 8x1 array.
        """
        fea
        return self.getFeature(self.env.perseus).reshape(-1)

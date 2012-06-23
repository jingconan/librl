from pybrain.rl.environments.mazes import MDPMazeTask
import sys
sys.path.append("..")
# from util import *
from util import debug, Expect
# from scipy import array

def TPNormalize(allowns, allowTP, state, uSize):
    '''if sum of tran prob is not 1, the rest tran prob will point to state itself'''
# --- Nomalize allowTP Method 2 ---
    allowns.append(state)
    for u in xrange(uSize):
        sTP = sum(allowTP[u])
        if  sTP < 1.0:
            allowTP[u].append(1.0 - sTP)
        else:
            allowTP[u].append(0)
    return allowns, allowTP

class RobotMotionTask(MDPMazeTask):
    '''This is a MDP Maze Task for Robot Motion Control.
    There are some goal states and some unsafe states.
    If the robot reach unsafe states, the reward(cost)
    will be 1. otherwise the reward will be zero. We want
    to reduce the probabiliy of robot falling into the
    unsafe region.
    '''
    def __init__(self, environment, **argv):
        MDPMazeTask.__init__(self, environment)
        self.reachGoalFlag = False
        self.senRange = argv['senRange']

    def reset(self):
        self.reachGoalFlag = False
        self.env.reset()

    @debug
    def getObservation(self):
        """the agent receive its
        1. position in the maze
        2. featureList for the state and for each possible action in the future.
           now featureList is [safety, process]
        so a typical observation will be a list with two elements. obs[0] is the position
        obs[1] is again a list with 4 elements, each element is the feature list for each
        possible actions.
        """
        obs = [self.env.perseus, self._getFeatureListC()]
        return obs

    @debug
    def performAction(self, action):
        super(RobotMotionTask, self).performAction(action)

    @debug
    def getReward(self):
        ''' compute and return the current reward (i.e.
        corresponding to the last action performed) '''
        # print 'RobotMotionTask::getReward'
        reward = 1 if self.env.bang else 0
        if self.env.perseus in self.env.goalStates: # FIXME be careful about type
            # print 'reach goal!!'
            self.env.reset()
            self.reachGoalFlag = True
            reward = 0
        return reward

    def _getFeatureListC(self):
        """Get Feature List for Current State"""
        return self._getFeatureList(self.env.perseus)


    @staticmethod
    def scale(x):
        """scale list x to [0, 1]"""
        min_x = min(x)
        rg = max(x) - min_x
        return [ ( v - min_x ) / rg for v in x]

    def _getFeatureList(self, x):
        """We can get features for each state. it may be
        distance to goal, safety degree. e.t.c"""
        allowns, allowTP, enable = self._GetAllowNSTP(x)
        DF = self.env.DF
        goalStates = self.env.goalStates

        # Calcualate Safety Degree
        nss = self._GetNSS(allowns, self.senRange) #FIXME
        es  = [ Expect(nss, pv) for pv in allowTP ]
        # safety = es
        safety = self.scale(es)
        # cnss = float( self._GetNSS([x], self.senRange)[0] )
        # safety = [esv - cnss for esv in es]

        # Calculate Expected Dist To Goal State
        CalDTG = lambda st: min([ DF(st, gs) for gs in goalStates])
        dtg = CalDTG(x)
        nsdtg = [ CalDTG(s) for s in allowns ] # next state dist to goal
        etg = [ Expect(nsdtg, pv) for pv in allowTP  ]
        # progress = [dtg - etgv for etgv in etg]
        progress = self.scale([dtg - etgv for etgv in etg])

        return zip(safety, progress)

    def _GetNSS(self, allowns, senRange):
        '''Get Next State Safety Degree '''
        isTrap = self.env._isTrap
        getMultiStepNS = self.env._GetMultiStepNS

        nss = []
        for s in allowns:
            if isTrap(s):
                nss.append( 0 )
            else:
                mns = getMultiStepNS(s, senRange)
                # print 'mns,', mns
                nss.append( sum([ ( not isTrap(st) ) for st in mns]) / ( len(mns) + 0.0 ))
        # print 'nss, ', nss
        return nss

    def _GetAllowNSTP(self, state):
        '''Get tranistion probability to allowable next step,
        allowns is the set of allowble next state
        allowTP is the transition probability under differnece action. it is a list of list,
        the 1st dimension = control size. the 2nd dimension = len(allowns)
        '''
        GetNS = self.env._GetNS
        OutBound = self.env._isOutBound
        uSize = self.env.numActions
        TP = self.env.TP

        ns = GetNS(state)
        l = len(ns)
        enable = [i for i in xrange(l) if not OutBound(ns[i])] # an indicator sequence. the value will be 1 if the corresponding state is enabled.
        allowns = [ns[i] for i in enable]
        allowTP = [ [ TP[u][j] for j in enable] for u in xrange(uSize)]
        allowns, allowTP = TPNormalize(allowns, allowTP, state, uSize)
        # print 'state, ', state
        # print 'allowns, ', allowns
        # print 'allowTP, ', allowTP

        assert( all( [abs( sum(tpu) - 1  ) < 1e-4 for tpu in allowTP] ) )
        assert(len(allowTP[i]) == len(allowns))
        return allowns, allowTP, enable

    ################################################################
    #  API added for enac     #
    ################################################################
    def isFinished(self):
        # print 'self.env.perseus', self.env.perseus, 'goalStates, ', self.env.goalStates
        # if self.env.perseus in self.env.goalStates:
        if self.reachGoalFlag:
            print 'task finished'
            return True
        else:
            return False

import unittest
from scipy import zeros
class RobotMotionTaskTestCase(unittest.TestCase):
    def setUp(self):
        print "In method", self._testMethodName
        from TrapMaze import TrapMaze
        iniState = (0, 0)
        gridSize = (5, 5)
        goalStates = [ (4, 4) ]
        unsafeStates = [ [2, 0], [3, 2], [2, 3] ]

# TP is transition probability
# The order is ENWS
        TP = [[0.7, 0.1, 0.1, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.1, 0.7, 0.1],
                [0.1, 0.1, 0.1, 0.7]]

        DF = lambda x, y: abs(x[0] - y[0]) + abs(x[1] - y[1])
        envMatrix = zeros(gridSize)
        envMatrix[zip(*unsafeStates)] = -1
        env = TrapMaze(envMatrix, iniState, goalStates, TP, DF)
        self.task = RobotMotionTask(env,
                senrange = 2)

    def test_GetAllowNSTP(self):
        pass

    def test_GetNSS(self):
        pass

    def test_getFeatureList(self):
        pass

    def test_getObservation(self):
        pass

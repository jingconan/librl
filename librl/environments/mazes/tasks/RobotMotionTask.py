from pybrain.rl.environments.mazes import MDPMazeTask
import sys
sys.path.append("..")
# from util import *
from librl.util import debug, Expect
from scipy import array

def TPNormalize(allowNextStates, allowTP, state, uSize):
    '''if sum of tran prob is not 1, the rest tran prob will point to state itself'''
# --- Nomalize allowTP Method 2 ---
    allowNextStates.append(state)
    for u in xrange(uSize):
        sTP = sum(allowTP[u])
        if  sTP < 1.0:
            allowTP[u].append(1.0 - sTP)
        else:
            allowTP[u].append(0)
    return allowNextStates, allowTP

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

        self.fea_memory = dict()

    def reset(self):
        self.reachGoalFlag = False
        self.env.reset()

    @debug
    def getObservation(self):
        """the agent receive its
        1. featureList for the state and for each possible action in the future.
           now featureList is [safety, process]
        if there are four possible controls. then agen will receive 8x1 array.
        """
        # obs = [self.env.perseus, self._getFeatureListC()]
        featureList = self._getFeatureList(self.env.perseus)
        return array(featureList).reshape(-1)

    @debug
    def performAction(self, action):
        super(RobotMotionTask, self).performAction(action)

    @debug
    def getReward(self):
        ''' compute and return the current reward (i.e.
        corresponding to the last action performed) '''
        # print 'RobotMotionTask::getReward'
        # reward = 1 if self.env.bang else 0
        reward = -1 if self.env.bang else 0
        # reward = -1 if self.env.bang else -0.001
        # reward = -100 if self.env.bang else -1
        if self.env.bang: self.env.reset()
        # print 'current position, ', self.env.perseus
        if self.env.perseus in self.env.goalStates: # FIXME be careful about type
            # print 'reach goal!!'
            self.env.reset()
            self.reachGoalFlag = True
            reward = 0
            # reward = 1000
            # reward = 10
            # reward = 100
        return reward

    @staticmethod
    def scale(x):
        """scale list x to [0, 1]"""
        min_x = min(x)
        rg = max(x) - min_x
        if rg == 0 :
            return [v*1.0/min_x for v in x]
        return [ ( v - min_x )*1.0/rg for v in x]

    def _getFeatureList(self, x):
        """We can get features for each state. it may be
        distance to goal, safety degree. e.t.c"""
        # cache the feature result, boost the speed.
        stored_fea = self.fea_memory.get(x, None)
        if stored_fea: return stored_fea

        nextStates, allowTP = self._GetAllowNextStates(x)
        DF = self.env.DF
        goalStates = self.env.goalStates

        # Calcualate Safety Degree
        nss = self._GetNSS(nextStates, self.senRange) #FIXME
        es  = [ Expect(nss, pv) for pv in allowTP ]
        safety = self.scale(es)

        # Calculate Expected Dist To Goal State
        CalDTG = lambda st: min([ DF(st, gs) for gs in goalStates])
        dtg = CalDTG(x)
        nsdtg = [ CalDTG(s) for s in nextStates ] # next state dist to goal
        etg = [ Expect(nsdtg, pv) for pv in allowTP  ]
        progress = self.scale([dtg - etgv for etgv in etg])

        fea = zip(safety, progress)
        self.fea_memory[x] = fea
        return fea
        # return zip(safety, progress)

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

    def _GetAllowNextStates(self, state):
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

        # the index of states in ns that are enabled.
        allowIndices = [i for i in xrange(l) if not OutBound(ns[i])]
        allowns = [ns[i] for i in allowIndices]
        allowTP = [ [ TP[u][j] for j in allowIndices] for u in xrange(uSize)]
        allowns, allowTP = TPNormalize(allowns, allowTP, state, uSize)

        assert( all( [abs( sum(tpu) - 1  ) < 1e-4 for tpu in allowTP] ) )
        assert(len(allowTP[i]) == len(allowns))
        return allowns, allowTP

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

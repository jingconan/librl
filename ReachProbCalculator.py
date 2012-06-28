#!/usr/bin/env python
import numpy as np
from numpy.linalg import solve
from time import clock as now
from util import *

class ReachProbCalculator:
    '''This Reachability Probability Calculator is only written for size (50, 50)'''
    def __init__(self, environment, task, agent):
        self.env = environment
        self.task = task
        self.agent = agent
        self.controlSize = self.env.numActions
        self.gridSize = self.env.mazeSize
        self.stateSize =  self.gridSize[0] * self.gridSize[1]
        self.iniState = self.env.startPos
        self.goalStates = self.env.goalStates
        self.ReachProbRes = []


    def __GetStateIdx(self, state):
        idx = state[0] * self.gridSize[1] + state[1]
        return idx

    def __IsSafe(self, state):
        return not self.env._isTrap(tuple(state))

    def __IsGoal(self, state):
        return ( tuple(state) in self.goalStates )

    def GetReachProb(self, theta):
        start = now()
        (A, b) = self.__GetCoeff(theta)
        x = solve(A, b)
        end = now()
        totalTime = end - start
        # print 'GetReachProb use time: ' + str(totalTime) + ' s'

        iniState = self.iniState
        iniIdx = self.__GetStateIdx(iniState)
        # print 'Reachability Probability of Initial State is: ' + str(x[iniIdx])
        self.ReachProbRes.append( float(x[iniIdx]) )
        return float( x[iniIdx] ), totalTime

    def GetActionProb(self, state, theta):
        feaList = self.task._getFeatureList(state)
        # print 'feaList, ', feaList
        return self.agent.policy.getActionProb(feaList, theta)

    def __GetCoeff(self, theta):
        A = np.zeros( (self.stateSize, self.stateSize) )
        b = np.zeros( (self.stateSize, ) )
        for x in range(self.gridSize[0]):
            for y in range(self.gridSize[1]):
                state = (x, y)
                i = self.__GetStateIdx( state )
                A[i, i] = 1
                if self.__IsSafe( state ) == 0:
                    b[i] = 0
                    continue
                if self.__IsGoal( state ) == 1:
                    b[i] = 1
                    continue

                ap = self.GetActionProb(state, theta)
                allowns, allowTP, enable = self.task._GetAllowNSTP(state)

                b[i] = 0
                # print 'ap', ap
                assert(abs( sum(ap) - 1) < 1e-4)
                for u in range(self.controlSize):
                    # print 'allowTP[u]', sum(allowTP[u])
                    assert( abs( sum( allowTP[u] ) - 1.0 ) < 1e-4 )
                    for (s, p) in zip(allowns, allowTP[u]):
                        sIdx = self.__GetStateIdx(s)
                        A[i, sIdx] -= ap[u] * p

        return (A, b)



if __name__ == "__main__":
    from main import *
    import settings
    import sys
    rp = reachProb.GetReachProb(GetActionProb, [float(sys.argv[1]),  float(sys.argv[2])])
    print 'reachprob: ', rp









from pybrain.rl.environments.mazes import Maze
from util import *


# GetNS = lambda x :[[x[0]+1, x[1]], [ x[0], x[1]+1 ], [ x[0]-1, x[1]  ], [ x[0], x[1]-1 ] ]
class TrapMaze(Maze):
    '''The difference between TrapMaze and Maze is that
    - in mazeTable, 1 means wall, -1 means trap.
    - when robot reaches a Wall(unsafeState), it will start over,
    - The robot is not fully controllable, it is a markov chain and described by transition probability.
    - the order of action is [N, E, W, S]
    '''
    trapFlag = -1
    wallFlag = 1
    def __init__(self, topology, startPos, goalStates, TP, DF, **args):
        assert(type(startPos) == types.TupleType)
        self.perseus = self.startPos = startPos
        self.initPos = [startPos]
        self.goalStates = goalStates
        self.TP = TP
        self.mazeSize = topology.shape
        self.mazeTable = topology
        self.setArgs(**args)
        self.bang = False
        TrapMaze.allActions = [Maze.N, Maze.E, Maze.W, Maze.S]
        self.numActions = len(TrapMaze.allActions)
        self.DF = DF

    def _isTrap(self, pos): return self.mazeTable[pos[0], pos[1]] == TrapMaze.trapFlag

    def _isWall(self, pos): return self.mazeTable[pos[0], pos[1]] == TrapMaze.wallFlag

    def _isOutBound(self, pos):
        return True if ( pos[0] >= self.mazeSize[0] or pos[1] >= self.mazeSize[1] or pos[0] < 0 or pos[1] < 0) else False

    def performAction(self, action):
        """TrapMaze is stochastic. When the control is E, the robot doen't necessarily
        goto the east direction, instead there is some transition probability to W, N, S, too.
        if the next position is out of the scene, the robot will not move. If the next position
        is a trap, the robot to go back to starting position and the self.bang flag is set."""
        realAction = GenRand(self.TP[action], self.allActions)
        tmp = self._moveInDir(self.perseus, realAction)
        if self._isOutBound(tmp) or self._isWall(tmp): # Short-Cricuit Effect
            self.bang = False
        elif self._isTrap(tmp):
            self.perseus, self.bang = self.startPos, True
            # print 'move to trap'
        else:
            self.perseus, self.bang = tmp, False
            # print 'move to: ',self.perseus

    def _GetNS(self, x):
        # return [[x[0]+1, x[1]], [ x[0], x[1]+1 ], [ x[0]-1, x[1]  ], [ x[0], x[1]-1 ] ]
        return [ (x[0]+a[0], x[1]+a[1]) for a in self.allActions]

    def _GetMultiStepNS(self, x, k):
        #FIXME only a stub
        DF = self.DF
        OutBound = self._isOutBound
        mns = []
        for a in range(x[0]- k, x[0]+ k):
            for b in range(x[1]-k, x[1]+k):
                state = [a, b]
                if DF(x, state) <= k and not OutBound(state):
                    mns.append(state)
        return mns

    def GetNSC(self):
        """Get the neighbor state for current state"""
        return self._GetNS(self.perseus)

    def GetMultiStepNSC(self, k):
        """Get multistep neighbor state for current state"""
        return self._GetMultiStepNS(self.perseus, k)

    def reset(self):
        self.bang = False
        self.perseus = self.startPos



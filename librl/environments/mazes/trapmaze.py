import scipy
import types
from pybrain.rl.environments.mazes import Maze

class TrapMaze(Maze):
    """The difference between TrapMaze and Maze is that
    - in mazeTable, 1 means wall, -1 means trap.
    - when robot reaches a Wall(unsafeState), it will start over,
    - The robot is not fully controllable, it is a markov chain and described
      by transition probability.
    - the order of action is [N, E, S, W]
    """
    # directions
    N = (-1, 0)
    S = (1, 0)
    E = (0, 1)
    W = (0, -1)

    WALL_FLAG = 1
    TRAP_FLAG = -1
    GOAL_FLAG = 2
    def __init__(self, topology, startPos, tranProb, **args):
        assert(type(startPos) == types.TupleType)
        self.perseus = self.startPos = startPos
        self.initPos = [startPos]
        self.tranProb = tranProb
        if (type(topology) == types.ListType):
            topology = scipy.array(topology)
        self.mazeSize = topology.shape
        self.mazeTable = topology
        self.setArgs(**args)
        self.bang = False
        self.allActions = [self.N, self.E, self.S, self.W]
        self.numActions = len(self.allActions)

    def isTrap(self, pos):
        return self.mazeTable[pos[0], pos[1]] == self.TRAP_FLAG

    def isWall(self, pos):
        return self.mazeTable[pos[0], pos[1]] == self.WALL_FLAG

    def isGoal(self, pos):
        return self.mazeTable[pos[0], pos[1]] == self.GOAL_FLAG

    def isOutBound(self, pos):
        return True if ( pos[0] >= self.mazeSize[0] or pos[1] >= self.mazeSize[1] or pos[0] < 0 or pos[1] < 0) else False

    def performAction(self, action):
        """TrapMaze is stochastic. When the control is E, the robot doen't necessarily
        goto the east direction, instead there is some transition probability to W, N, S, too.
        if the next position is out of the scene, the robot will not move. If the next position
        is a trap, the robot to go back to starting position and the self.bang flag is set."""
        assert action >= 0 and action < self.numActions
        actions = range(self.numActions)
        realActionIndex = scipy.random.choice(actions,
                                              p=self.tranProb[action])
        realAction = self.allActions[realActionIndex]
        nextPos = self._moveInDir(self.perseus, realAction)

        if self.isOutBound(nextPos) or self.isWall(nextPos): # Short-Cricuit Effect
            # position (perseus) is not changed.
            self.bang = True
        elif self.isTrap(nextPos):
            self.perseus = self.startPos
            self.bang = True
        else:
            self.perseus = nextPos
            self.bang = False

    def reset(self):
        self.bang = False
        self.perseus = self.startPos

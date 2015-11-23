import sys
sys.path.append("..")
from RobotMotionTask import RobotMotionTask
from scipy import array


class SimpleTemporalLogic(RobotMotionTask):
    def hash(self, x):
        s = self.env.mazeSize
        return x[0] + x[1] * s[0]

    def getObservation(self):
       return array([self.hash(self.env.perseus)])

    def getReward(self):
        ''' compute and return the current reward (i.e.
        corresponding to the last action performed) '''
        # print 'RobotMotionTask::getReward'
        reward = -1 if self.env.bang else 0
        if self.env.bang: self.env.reset()
        if self.env.perseus in self.env.goalStates: # FIXME be careful about type
            # print 'reach goal!!'
            self.env.reset()
            self.reachGoalFlag = True
            # reward = 0
            reward = 10
        return reward


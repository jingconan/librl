#!/usr/bin/env python
"""
This file is the simulation of LSTD Actor Critic algorithm.
The output will be a trace file.
"""
__author__ = 'Jing Conan Wang, Boston University, wangjing@bu.edu'

import sys
sys.path.insert(0, "..")

from Process import Process
from ReachProbCalculator import ReachProbCalculator
class LSTDACProcess(Process):
    def _learner(self):
        from learners import LSTDACLearner
        return LSTDACLearner(lamb = 0.9, c = 10, D=20)

    def _init_trace(self):
        self.trace = dict(
                    ep=[], # episode number
                    reward=[], # reward of this epsisode
                    it=[], # total iteration number
                    theta0=[], # first value of theta
                    theta1=[], # second value of theta
                    )
        # more detailed record for theta value
        self.th_trace = dict(
                theta0=[],
                theta1=[],
                it=[],
                )
        self.reachProb = ReachProbCalculator(self.env, self.task, self.agent)
        self.epis_num = 0

    def export_trace(self):
        from util import WriteTrace
        WriteTrace(self.trace, './res/lstdac.tr')
        WriteTrace(self.th_trace, './res/lstdac_theta.tr')

    def check(self):
        if self.j == 1e6:
            return True

        if self.j % 100 == 0:
            print 'iter: [', self.j, '] theta value: [%f, %f]'%tuple(self.policy.theta)
            self.th_trace['theta0'].append(self.policy.theta[0])
            self.th_trace['theta1'].append(self.policy.theta[1])
            self.th_trace['it'].append(self.j)

        if self.j % 3e3 == 0:
            rp, time = self.reachProb.GetReachProb(self.policy.theta)
            print 'rp, ', rp, 'time, ', time
            if rp > 0.5:
                return True

        if self.task.reachGoalFlag:
            self.epis_num += 1
            # if i == 5e3: break
            self.trace['theta0'].append(self.policy.theta[0])
            self.trace['theta1'].append(self.policy.theta[1])
            self.trace['ep'].append(self.epis_num)
            self.trace['reward'].append(self.r)
            self.trace['it'].append(self.j)
            print '[%i]reach goal, reward'%(self.epis_num), self.r
            self.r = 0
            self.task.reset()
            self.learner.resetStepSize()

        return False
if __name__ == "__main__":
    from time import clock
    start = clock()
    proc = LSTDACProcess()
    proc.loop_exception_handle()
    end = clock()
    print 'total time is %i'%(end-start)

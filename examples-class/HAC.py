#!/usr/bin/env python
"""
This file is the simulation of Hessian Actor Critic algorithm.
The output will be a trace file.
"""
__author__ = 'Jing Conan Wang, Boston University, wangjing@bu.edu'

import sys
sys.path.insert(0, "..")

from LSTDAC import LSTDACProcess
class HACProcess(LSTDACProcess):
    def _learner(self):
        from learners import HACLearner
        return HACLearner(lamb = 0.9, c = 10, D=20)

    def export_trace(self):
        from util import WriteTrace
        WriteTrace(self.trace, './res/hac.tr')
        WriteTrace(self.th_trace, './res/hac_theta.tr')

if __name__ == "__main__":
    from time import clock
    start = clock()
    proc = HACProcess()
    proc.loop_exception_handle()
    end = clock()
    print 'total time is %i'%(end-start)

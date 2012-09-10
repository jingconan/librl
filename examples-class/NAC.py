#!/usr/bin/env python
"""
This file is the simulation of Natural Actor Critic algorithm.
The output will be a trace file.
"""
__author__ = 'Jing Conan Wang, Boston University, wangjing@bu.edu'

import sys
sys.path.insert(0, "..")

from LSTDAC import LSTDACProcess
class NACProcess(LSTDACProcess):
    def _learner(self):
        from learners.NAC import NAC
        return NAC(lamb = 1, c = 0.8, D=2)

    def export_trace(self):
        from util import WriteTrace
        WriteTrace(self.trace, './res/nac.tr')
        WriteTrace(self.th_trace, './res/nac_theta.tr')

if __name__ == "__main__":
    from time import clock
    start = clock()
    proc = NACProcess()
    proc.loop_exception_handle()
    end = clock()
    print 'total time is %i'%(end-start)

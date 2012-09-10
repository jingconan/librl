#!/usr/bin/env python
""" Use Customized Episodic Natural Actor Critic Method to solve Temporal Logic Problem
"""
__author__ = 'Jing Conan Wang, Boston University, wangjing@bu.edu'

from LSTDAC import LSTDACProcess
class HACProcess(LSTDACProcess):
    def _learner(self):
        from learners.CENACLearner import CENAC
        return CENAC()

    def _agent(self):
        from agents import ExplorerLearningAgent
        self.policy = self._policy()
        self.learner = self._learner()
        return ExplorerLearningAgent(self.policy, self.learner)

    def _experiment(self):
        from pybrain.rl.experiments import EpisodicExperiment
        self.task = self._task()
        self.agent = self._agent()
        return EpisodicExperiment(self.task, self.agent)



    def export_trace(self):
        from util import WriteTrace
        WriteTrace(self.trace, './res/hac.tr')
        WriteTrace(self.th_trace, './res/hac_theta.tr')

    def check(self):
        if self.j >= 1e6:
            return True
        # if j >= 1e5: break
        # if j >= 2.5 * 1e5: break

        # print 'theta, [%f, %f]'%tuple(learner.theta)
        print 'iter: [', self.j, '] theta value: [%f, %f]'%tuple(self.policy.theta)
        self.th_trace['theta0'].append(self.policy.theta[0])
        self.th_trace['theta1'].append(self.policy.theta[1])
        self.th_trace['it'].append(self.j)

        if self.task.reachGoalFlag:
            self.epis_num += 1
            # if i == 5e3: break
            self.trace['ep'].append(self.epis_num)
            self.trace['reward'].append(self.r)
            self.trace['it'].append(self.j)
            self.trace['theta0'].append(self.policy.theta[0])
            self.trace['theta1'].append(self.policy.theta[1])
            print '[%i]reach goal, reward'%(self.epis_num), self.r

        return False


    def loop(self):
        experiment = self._experiment()
        self._init_trace()
        self.r = 0
        self.j = -1
        while True:
            rewards = experiment.doEpisodes(5)
            # import pdb;pdb.set_trace()
            self.r = sum([sum(rset) for rset in rewards ])
            self.j += sum([len(rset) for rset in rewards ])

            self.agent.learn()
            if self.check():
                break
        self.export_trace()



if __name__ == "__main__":
    from time import clock
    start = clock()
    proc = HACProcess()
    proc.loop_exception_handle()
    end = clock()
    print 'total time is %i'%(end-start)

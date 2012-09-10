#!/usr/bin/env python
"""
This file is the simulation of Hessian Actor Critic algorithm.
The output will be a trace file.
"""
__author__ = 'Jing Conan Wang, Boston University, wangjing@bu.edu'

import sys
sys.path.insert(0, "..")

##############################################
##    Import Usefule Modules and Functins   ##
##############################################
from scipy import zeros
from pybrain.rl.experiments import Experiment
from policy import BoltzmanPolicy
from environments import TrapMaze
from task import RobotMotionTask, SimpleTemporalLogic
from learners import HACLearner
from agents import ACAgent

###############################
##       Parameters          ##
###############################
# import global parameters
from problem_settings import gridSize, unsafeStates, iniState, goalStates, TP, DF, senRange
from problem_settings import T, iniTheta

# Create environment
# Add unsafe states
envMatrix = zeros(gridSize)
envMatrix[zip(*unsafeStates)] = -1
env = TrapMaze(envMatrix, iniState, goalStates, TP, DF)

# Create task
task = RobotMotionTask(env, senRange=senRange)
# task = SimpleTemporalLogic(env, senRange=senRange)

policy = BoltzmanPolicy(feaDim = 2, numActions = 4, T = T, iniTheta=iniTheta)
# learner = HACLearner(lamb = 0.9, c = 0.8, D=2)
# learner = HACLearner(lamb = 0.9, c = 5, D=10)
# learner = HACLearner(lamb = 0.9, c = 1.2, D=4)
learner = HACLearner(lamb = 0.9, c = 10, D=20)
# learner = HACLearner(lamb = 0.9, c = 5, D=10)
# learner = HACLearner(lamb = 0.9, c = 2, D=10)
agent = ACAgent(policy, learner, sdim=8, adim=1)
experiment = Experiment(task, agent)

from ReachProbCalculator import *
reachProb = ReachProbCalculator(env, task, agent)

def main():
    i = -1
    # trace record reward of each episode
    trace = dict(ep=[], # episode number
            reward=[], # reward of this epsisode
            it=[], # total iteration number
            theta0=[], # first value of theta
            theta1=[], # second value of theta
            )

    # more detailed record for theta value
    th_trace = dict(
            theta0=[],
            theta1=[],
            it=[],
            )

    r = 0
    j = -1
    try:
        while True:
            j += 1
            reward = experiment._oneInteraction()
            r += reward
            agent.learn()
            if j == 1e6: break
            # if j == 1e5: break
            if j % 100 == 0:
                print 'iter: [', j, '] theta value: [%f, %f]'%tuple(policy.theta)
                th_trace['theta0'].append(policy.theta[0])
                th_trace['theta1'].append(policy.theta[1])
                th_trace['it'].append(j)

            if j % 3e3 == 0:
                rp, time = reachProb.GetReachProb(policy.theta)
                print 'rp, ', rp, 'time, ', time
                if rp > 0.5:
                    break

            if task.reachGoalFlag:
                i += 1
                # if i == 5e3: break
                trace['theta0'].append(policy.theta[0])
                trace['theta1'].append(policy.theta[1])
                trace['ep'].append(i)
                trace['reward'].append(r)
                trace['it'].append(j)
                print '[%i]reach goal, reward'%(i), r
                r = 0
                task.reset()
                learner.resetStepSize()
                continue
    except KeyboardInterrupt:
        pass
# except Exception as e:
        # pass

    from util import WriteTrace
    WriteTrace(trace, 'hac.tr')
    WriteTrace(th_trace, 'hac_theta.tr')

if __name__ == "__main__":
    from time import clock
    start = clock()
    main()
    end = clock()
    print 'total time is %i'%(end-start)

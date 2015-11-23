#!/usr/bin/env python

__author__ = 'Jing Conan Wang, Boston University, wangjing@bu.edu'
'''This example demostrates how to use LSTD Actor-Critic to solve Robot Motion Control Problem in pybrain framework'''

from scipy import zeros, array
import pylab, time

from pybrain.rl.environments.mazes import Maze
# from pybrain.rl.agents.linearfa import LinearFA_Agent
# from pybrain.rl.learners.valuebased.linearfa import LSTDQLambda
from pybrain.rl.experiments import *
from task import *
from environments import *
from learners import *
from agents import *
from policy import *
from ReachProbCalculator import *

# import global parameters
import settings
from settings import gridSize, unsafeStates, goalStates, TP, iniState, VISUAL, DF, iniTheta, lamb, c, D, showInterval, T, senRange, uSize, hessianThetaTh

# Create environment
# Add unsafe states
envMatrix = zeros(gridSize)
envMatrix[zip(*unsafeStates)] = -1
env = TrapMaze(envMatrix, iniState, goalStates, TP, DF)

# Create task
task = RobotMotionTask(env, senRange=senRange)

sDim = 2
aDim = 1
# Create learner
# learner = LSTDQLambda(4, 1)
# learner = LSTDACLearner(aDim, iniTheta,
        # lamb=lamb, c=c, D=D)
learner = TDLearner(aDim, iniTheta,
        lamb=lamb, c=c, D=D)
#
# learner = HessianACLearner(aDim, iniTheta,
        # uSize=uSize, gridSize=gridSize, lamb=lamb,
        # c=c, D=D, hessianThetaTh=hessianThetaTh)

# Create policy
# T = 50
policy = BoltzmanPolicy(T)

# Create agent
# agent = LinearFA_Agent(learner)
agent = LSTDACAgent(policy, learner, sDim, aDim)
# agent = HessianACAgent(policy, learner, sDim, aDim)

# create experiment
# experiment = ContinuousExperiment(task, agent)
experiment = Experiment(task, agent)

# VISUAL = True
# prepare plotting
if VISUAL:
    pylab.gray()
    pylab.ion()

cFlag = dict(gs=-2, robot=2, trap=-1, normal=0)
visEnvMat = envMatrix
for gs in goalStates:
    visEnvMat[tuple(gs)] = cFlag['gs']
lastPos = None

# experiment.doInteractionsAndLearn(100)
reachProb = ReachProbCalculator(env, task, agent)

# for i in range(10000):
# TRACE = False
trace = dict(i=[], rp=[])
from util import WriteTrace
if __name__ == "__main__":
    i = -1
    # try:
    while True:
        i += 1
        if i == 1e4:
            break
        experiment.doInteractions(1)
        agent.learn()
        if task.reachGoalFlag:
            print 'reach goal'
            task.reset()
            continue

        if i % showInterval == 0: print 'theta value, [%f, %f]'%tuple(agent.learner.theta)
        # if settings.CAL_EXACT_PROB and i % settings.reachProbInterval == 0:
            # rp, t = reachProb.GetReachProb(agent.learner.theta)
            # print 'iter: [%d] reachProb: %f,  aveCost: %f it takes, %f seconds' %(i, rp, 1.0 / rp - 1, t)
            # trace['i'].append(i)
            # trace['rp'].append(rp)

        if lastPos is not None: visEnvMat[lastPos] = cFlag['normal']
        visEnvMat[env.perseus] = cFlag['robot']; lastPos = env.perseus

        if VISUAL:
            pylab.pcolor(visEnvMat)
            pylab.draw()
            # pause(0.1)
            # pylab.pause(0.1)
            time.sleep(0.1)
    # except KeyboardInterrupt as e:
        # print e
        # import traceback
        # traceback.print_stack()
        # pass
    # except Exception as e:
        # print e
        # pass

    # WriteTrace(trace, 'rp_rec.txt')


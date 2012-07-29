#!/usr/bin/env python
__author__ = 'Jing Conan Wang, Boston University, wangjing@bu.edu'

import sys
sys.path.insert(0, "..")
from scipy import zeros

from pybrain.rl.experiments import Experiment

from policy import BoltzmanPolicy
from environments import TrapMaze
from task import RobotMotionTask, SimpleTemporalLogic
from learners import LSTDACLearner, TDLearner
from agents import ACAgent

# import global parameters
from problem_settings import gridSize, unsafeStates, iniState, goalStates, TP, DF, senRange
from problem_settings import iniTheta, T

# Create environment
# Add unsafe states
envMatrix = zeros(gridSize)
envMatrix[zip(*unsafeStates)] = -1
env = TrapMaze(envMatrix, iniState, goalStates, TP, DF)

# Create task
task = RobotMotionTask(env, senRange=senRange)
# task = SimpleTemporalLogic(env, senRange=senRange)

# policy = BoltzmanPolicy(feaDim = 2, numActions = 4, T = 10, iniTheta=[0, 0])
policy = BoltzmanPolicy(feaDim = 2, numActions = 4, T = T, iniTheta=iniTheta)
# learner = LSTDACLearner(lamb = 1, c = 0.8, D=2)
learner = TDLearner(lamb = 0.9, c = 0.8, D=2)
# learner = TDLearner(lamb = 0.8, c = 0.8, D=2)
# learner = TDLearner(lamb = 0.9, c = 5, D=10)
# agent = ExplorerLearningAgent(policy, learner)
agent = ACAgent(policy, learner, sdim=8, adim=1)
experiment = Experiment(task, agent)

i = -1
# trace = dict(epsiode=[], rp=[])
th_trace = dict(theta0=[], theta1=[], it=[])

trace = dict(ep=[], reward=[], it=[],
        theta0=[], theta1=[])
r = 0
j = -1
try:
    while True:
        j += 1
        reward = experiment._oneInteraction()
        r += reward
        agent.learn()
        if j == 1e5: break
        if j % 100 == 0:
            print 'theta value: [%f, %f]'%tuple(policy.theta)
            th_trace['theta0'].append(policy.theta[0])
            th_trace['theta1'].append(policy.theta[1])
            th_trace['it'].append(j)

        if task.reachGoalFlag:
            i += 1
            if i == 5e3: break
            trace['ep'].append(i)
            trace['reward'].append(r)
            trace['theta0'].append(policy.theta[0])
            trace['theta1'].append(policy.theta[1])
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
WriteTrace(trace, 'tdac.tr')
WriteTrace(th_trace, 'tdac_theta.tr')

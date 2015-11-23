#!/usr/bin/env python
""" Use SARSA method to solve Temporal Logic Problem
"""
__author__ = 'Jing Conan Wang, Boston University, wangjing@bu.edu'

import sys
sys.path.append("..")
from scipy import zeros

from pybrain.rl.learners.valuebased.sarsa import SARSA
from pybrain.rl.agents.learning import LearningAgent
from pybrain.rl.learners.valuebased import ActionValueTable
from pybrain.rl.experiments import Experiment

from policy import BoltzmanPolicy
from environments import TrapMaze
from task import RobotMotionTask, SimpleTemporalLogic
from pybrain.rl.learners.valuebased.linearfa import Q_LinFA
from pybrain.rl.agents.linearfa import LinearFA_Agent
# import pylab, time

# import global parameters
from problem_settings import gridSize, unsafeStates, iniState, goalStates, TP, DF, senRange

# Create environment
# Add unsafe states
envMatrix = zeros(gridSize)
envMatrix[zip(*unsafeStates)] = -1
env = TrapMaze(envMatrix, iniState, goalStates, TP, DF)

# Create task
# task = RobotMotionTask(env, senRange=senRange)
task = SimpleTemporalLogic(env, senRange=senRange)


# create value table and initialize with ones
# table = ActionValueTable(gridSize[0]*gridSize[1], 4)
# table.initialize(1.)

# policy = BoltzmanPolicy(feaDim = 2, numActions = 4, T = 100)

# Create learner
learner = Q_LinFA(4, 1)

# create agent
# agent = LearningAgent(policy, learner)
# agent = LearningAgent(table, learner)
agent = LinearFA_Agent(learner)

# create experiment
experiment = Experiment(task, agent)

# prepare plotting
# import pylab
# pylab.gray()
# pylab.ion()
# from time import sleep

cFlag = dict(gs=-2, robot=2, trap=-1, normal=0)
visEnvMat = envMatrix
for gs in goalStates:
    visEnvMat[tuple(gs)] = cFlag['gs']
lastPos = None

i = -1
trace = dict(epsiode=[], rp=[])
r = 0
try:
    while True:
        reward = experiment._oneInteraction()
        r += reward
        agent.learn()

        # pylab.pcolor(table.params.reshape(gridSize[0]*gridSize[1],4).max(1).reshape(gridSize))
        # pylab.draw()
        # pylab.show()
        # sleep(0.1)

        # if lastPos is not None: visEnvMat[lastPos] = cFlag['normal']
        # visEnvMat[env.perseus] = cFlag['robot']; lastPos = env.perseus
        # lastPos = env.perseus

        # pylab.pcolor(visEnvMat)
        # pylab.draw()
        # pylab.show()
        # sleep(0.1)


        if task.reachGoalFlag:
            i += 1
            if i == 5e3: break
            trace['epsiode'].append(i)
            trace['rp'].append(r)
            print '[%i]reach goal, reward'%(i), r
            r = 0
            task.reset()
            continue
# except KeyboardInterrupt:
except:
    pass

from util import WriteTrace
WriteTrace(trace, 'rec.tr')



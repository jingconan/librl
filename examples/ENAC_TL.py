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
from task import RobotMotionTask
# import pylab, time

# import global parameters
from problem_settings import gridSize, unsafeStates, iniState, goalStates, TP, DF, senRange

# Create environment
# Add unsafe states
envMatrix = zeros(gridSize)
envMatrix[zip(*unsafeStates)] = -1
env = TrapMaze(envMatrix, iniState, goalStates, TP, DF)

# Create task
task = RobotMotionTask(env, senRange=senRange)

# create value table and initialize with ones
table = ActionValueTable(gridSize[0]*gridSize[1], 4)
table.initialize(1.)
# policy = BoltzmanPolicy(feaDim = 2, numActions = 4, T = 100)

# Create learner
learner = SARSA()

# create agent
agent = LearningAgent(policy, learner)

# create experiment
experiment = Experiment(task, agent)

i = -1
while True:
    i += 1
    experiment.doInteractions(1)
    agent.learn()
    if task.reachGoalFlag:
        print 'reach goal'
        task.reset()
        continue

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

# Create environment
# Add unsafe states
envMatrix = zeros(gridSize)
envMatrix[zip(*unsafeStates)] = -1
env = TrapMaze(envMatrix, iniState, goalStates, TP, DF)

# Create task
task = RobotMotionTask(env, senRange=senRange)
# task = SimpleTemporalLogic(env, senRange=senRange)

policy = BoltzmanPolicy(feaDim = 2, numActions = 4, T = 100)
learner = LSTDACLearner(lamb = 1, c = 0.8, D=2)
# learner = TDLearner(lamb = 1, c = 0.8, D=2)
# agent = ExplorerLearningAgent(policy, learner)
agent = ACAgent(policy, learner, sdim=8, adim=1)
experiment = Experiment(task, agent)

i = -1
trace = dict(epsiode=[], rp=[])
r = 0
while True:
    reward = experiment._oneInteraction()
    r += reward
    agent.learn()

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
# except Exception as e:
    # pass

from util import WriteTrace
WriteTrace(trace, 'enac_rec.tr')



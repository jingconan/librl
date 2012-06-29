#!/usr/bin/env python
""" Use SARSA method to solve Temporal Logic Problem
"""
__author__ = 'Jing Conan Wang, Boston University, wangjing@bu.edu'

import sys
sys.path.insert(0, "..")
from scipy import zeros

from pybrain.tools.example_tools import ExTools

from pybrain.rl.learners.valuebased.sarsa import SARSA
from pybrain.rl.agents.learning import LearningAgent
from pybrain.rl.learners.valuebased import ActionValueTable
from pybrain.rl.experiments import Experiment, EpisodicExperiment
# from pybrain.rl.learners.directsearch import ENAC
from learners.CENACLearner import CENAC

from agents import ExplorerLearningAgent
from policy import BoltzmanPolicy
from environments import TrapMaze
from task import RobotMotionTask, SimpleTemporalLogic
from pybrain.rl.learners.valuebased.linearfa import QLambda_LinFA
from pybrain.rl.agents.linearfa import LinearFA_Agent
from pybrain.rl.explorers.discrete import BoltzmannExplorer
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
# task = SimpleTemporalLogic(env, senRange=senRange)

# policy = BoltzmanPolicy(feaDim = 2, numActions = 4, T = 100)
policy = BoltzmanPolicy(feaDim = 2, numActions = 4, T = 10, iniTheta=[0, 0])
explorer = BoltzmannExplorer()

# learner = ENAC()
learner = CENAC()

# create agent
agent = ExplorerLearningAgent(policy, learner, explorer)

# create experiment
# experiment = Experiment(task, agent)
experiment = EpisodicExperiment(task, agent)


i = -1
trace = dict(ep=[], reward=[], it=[],
        theta0=[], theta1=[])
th_trace = dict(theta0=[], theta1=[], it=[])

r = 0
j = 0
try:
    while True:
        j += 1
        # reward = experiment._oneInteraction()
        rewards = experiment.doEpisodes(1)
        r = sum(rewards[0])
        j += len(rewards[0])
        agent.learn()

        if j >= 1e6: break

        print 'theta, [%f, %f]'%tuple(learner.theta)
        th_trace['theta0'].append(policy.theta[0])
        th_trace['theta1'].append(policy.theta[1])
        th_trace['it'].append(j)

        if task.reachGoalFlag:
            i += 1
            if i == 5e3: break
            trace['ep'].append(i)
            trace['reward'].append(r)
            trace['it'].append(j)
            trace['theta0'].append(policy.theta[0])
            trace['theta1'].append(policy.theta[1])
            print '[%i]reach goal, reward'%(i), r
            continue
except KeyboardInterrupt:
    pass
# except Exception as e:
    # pass

from util import WriteTrace
WriteTrace(trace, 'enac.tr')
WriteTrace(th_trace, 'enac_theta.tr')



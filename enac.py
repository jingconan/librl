#!/usr/bin/env python
"""Test for Natural Gradient Algorithm"""

from scipy import zeros, array
import pylab

from pybrain.tools.example_tools import ExTools
from pybrain.tools.shortcuts import buildNetwork
from pybrain.rl.environments.mazes import Maze
from pybrain.rl.experiments import *
# from pybrain.rl.learners import ENAC
# from pybrain.rl.agents import LearningAgent

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

# policy = BoltzmanPolicy(T)
policy = BoltzmanPolicy(feaDim = 2, numActions = 4, T = 10, iniTheta=[0, 0])
# net = buildNetwork(4, 1, bias=False)
learner = ENAC(iniTheta=iniTheta, learningRate=0.1)


agent = LSTDACAgent(policy, learner,  sDim, aDim)

reachProb = ReachProbCalculator(env, task, agent)

if __name__ == "__main__":
    epsiodNum = 1000
    for i in xrange(epsiodNum):
        experiment = EpisodicExperiment(task, agent)
        all_rewards = experiment.doEpisodes(1)
        agent.learn()
        print 'theta value, ', learner.theta
        rp, t = reachProb.GetReachProb(agent.learner.theta)
        print 'iter: [%d] reachProb: %f,  aveCost: %f it takes, %f seconds' %(i, rp, 1.0 / rp - 1, t)





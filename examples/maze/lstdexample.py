#!/usr/bin/env python
__author__ = 'Jing Conan Wang, Boston University, wangjing@bu.edu'

from scipy import zeros

from collections import defaultdict

from pybrain.rl.experiments import Experiment
import librl
from librl.policies import BoltzmanPolicy, PolicyFeatureModule
from librl.environments.mazes.trapmaze import TrapMaze
from librl.environments.mazes.tasks.robottask import RobotMotionAvgRewardTask
from librl.experiments import *
from librl.learners import *
from librl.agents.actorcriticagent import ActorCriticAgent
from librl.util import WriteTrace, cPrint

# import global parameters
from problem_settings import gridSize, unsafeStates, iniState, goalStates, TP, DF, senRange
from problem_settings import iniTheta, T

# Create environment
# Add unsafe states
envMatrix = zeros(gridSize)
envMatrix[zip(*unsafeStates)] = TrapMaze.TRAP_FLAG
envMatrix[zip(*goalStates)] = TrapMaze.GOAL_FLAG
env = TrapMaze(envMatrix, iniState, TP)

# Create task
task = RobotMotionAvgRewardTask(env, senRange)
task.GOAL_REWARD = 1
task.TRAP_REWARD = -1


sessionNumber = 1000
sessionSize = 100

feaDim = 8
numActions = 4

policy = BoltzmanPolicy(numActions, T, iniTheta)
module = PolicyFeatureModule(policy, 'policywrapper')
#  learner = TDLearner(module=module, 0.9, 8, 1)
learner = LSTDLearner(module=module,
                      cssinitial=0.1,
                      cssdecay=1000,
                      assinitial=0.01,
                      assdecay=1000, # ass means actor steps size
                      rdecay=0.95,
                      maxcriticnorm=10000, # maximum critic norm
                      tracestepsize=0.9 # stepsize of trace
                      )

agent = ActorCriticAgent(learner, sdim=feaDim, adim=1, batch=True)
experiment = SessionExperiment(task, agent, policy=policy, batch=True)

try:
    experiment.doSessionsAndPrint(sessionNumber=sessionNumber,
                                  sessionSize=sessionSize)
except KeyboardInterrupt:
    pass

#!/usr/bin/env python
__author__ = 'Jing Conan Wang, Boston University, wangjing@bu.edu'

from scipy import zeros

from collections import defaultdict

from pybrain.rl.experiments import Experiment
import librl
from librl.policies import BoltzmanPolicy, PolicyValueFeatureModule
from librl.environments.mazes.trapmaze import TrapMaze
from librl.environments.mazes.tasks.robottask import RobotMotionAvgRewardTask
from librl.experiments import *
from librl.learners.bsgl import *
from librl.agents.actorcriticagent import ActorCriticAgent
from librl.util import cPrint

# import global parameters
from problem_settings import gridSize, unsafeStates, iniState, goalStates, TP, DF, senRange
from problem_settings import iniTheta, T

#####################
#    Parameters     #
#####################
#  learnerClass = BSGLRegularGradientActorCriticLearner
#  learnerClass = BSGLFisherInfoActorCriticLearner
learnerClass = BSGLAdvParamActorCriticLearner
#  learnerClass = BSGLAdvParamFisherInfoActorCriticLearner


sessionNumber = 1000
sessionSize = 100

feaDim = 8
numActions = 4

# Create environment
# Add unsafe states
envMatrix = zeros(gridSize)
envMatrix[zip(*unsafeStates)] = TrapMaze.TRAP_FLAG
envMatrix[zip(*goalStates)] = TrapMaze.GOAL_FLAG
env = TrapMaze(envMatrix, iniState, TP)

# Create task
task = RobotMotionAvgRewardTask(env, senRange)
task.GOAL_REWARD = 10
task.TRAP_REWARD = -1

policy = BoltzmanPolicy(numActions, T, iniTheta)
featureModule = PolicyValueFeatureModule(policy, 'bsglpolicywrapper')
learner = learnerClass(cssinitial=0.1,
                       cssdecay=1000, # css means critic step size
                       assinitial=0.001,
                       assdecay=10000, # ass means actor steps size
                       rdecay=0.95, # reward decay weight
                       #  parambound=None # bound for the parameters
                       parambound=[[-50, 150], [-50, 50]], # bound for the parameters
                       maxcriticnorm=1000000,
                       module=featureModule
                       )

agent = ActorCriticAgent(learner, sdim=feaDim, adim=1, batch=True)
experiment = SessionExperiment(task, agent, policy=policy, batch=True)

try:
    experiment.doSessionsAndPrint(sessionNumber=sessionNumber,
                                  sessionSize=sessionSize)
except KeyboardInterrupt:
    pass

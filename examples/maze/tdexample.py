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
from librl.learners.td import TDLearner
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
task.GOAL_REWARD = 10
task.TRAP_REWARD = -1


sessionNumber = 1000
sessionSize = 1000

feaDim = 8
numActions = 4

policy = BoltzmanPolicy(numActions, T, iniTheta)
module = PolicyFeatureModule(policy, 'policywrapper')
#  learner = TDLearner(module=module, 0.9, 8, 1)
learner = TDLearner(module=module,
                    cssinitial=0.1,
                    cssdecay=1000,
                    assinitial=0.01,
                    assdecay=1000, # ass means actor steps size
                    rdecay=0.95,
                    maxcriticnorm=10000, # maximum critic norm
                    tracestepsize=0.9 # stepsize of trace
                    )

agent = ActorCriticAgent(learner, sdim=feaDim, adim=1)
experiment = Experiment(task, agent)

#####################
#    Parameters     #
#####################
ITER_NUM = 500000
#  LEARN_INTERVAL = 10
LEARN_INTERVAL = 1
OUTPUT_FILENAME = 'tdac.tr'

trace = defaultdict(list)

experiment = SessionExperiment(task, agent, policy=policy, batch=True)

try:
    experiment.doSessionsAndPrint(sessionNumber=sessionNumber,
                                  sessionSize=sessionSize)
except KeyboardInterrupt:
    pass

#  def loop():
#      for i in xrange(ITER_NUM):
#          reward = 0
#          for j in xrange(LEARN_INTERVAL):
#              reward += experiment._oneInteraction()
#          agent.learn()

        # periodically reset stepsize to increase learning speed.
#          if i % 1000 == 0:
#              learner.resetStepSize()
#          cPrint(iteration=i, th0=policy.theta[0], th1=policy.theta[1],
#                 reward=reward)
#  try:
#      loop()
#  except KeyboardInterrupt:
#      pass

#  WriteTrace(trace, OUTPUT_FILENAME)

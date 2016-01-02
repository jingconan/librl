#!/usr/bin/env python
__author__ = 'Jing Conan Wang, Boston University, wangjing@bu.edu'

from scipy import zeros

from collections import defaultdict

from pybrain.rl.experiments import Experiment
import librl
from librl.policies import BoltzmanPolicy
from librl.environments.mazes.trapmaze import TrapMaze
from librl.environments.mazes.tasks.robottask import RobotMotionAvgRewardTask
from librl.learners.bsgl import BSGLRegularGradientActorCriticLearner, BSGLFisherInfoActorCriticLearner
from librl.agents.actorcriticagent import ActorCriticAgent
from librl.util import cPrint

# import global parameters
from problem_settings import gridSize, unsafeStates, iniState, goalStates, TP, DF, senRange
from problem_settings import iniTheta, T

#####################
#    Parameters     #
#####################
ITER_NUM = 500000
#  LEARN_INTERVAL = 10
LEARN_INTERVAL = 1
#  learnerClass = BSGLRegularGradientActorCriticLearner
learnerClass = BSGLFisherInfoActorCriticLearner


# Create environment
# Add unsafe states
envMatrix = zeros(gridSize)
envMatrix[zip(*unsafeStates)] = TrapMaze.TRAP_FLAG
envMatrix[zip(*goalStates)] = TrapMaze.GOAL_FLAG
env = TrapMaze(envMatrix, iniState, TP)

# Create task
task = RobotMotionAvgRewardTask(env, senRange)
task.GOAL_REWARD = 100
task.TRAP_REWARD = 0

policy = BoltzmanPolicy(4, T, iniTheta)
learner = learnerClass(policy=policy,
                       cssinitial=0.1,
                       cssdecay=1000, # css means critic step size
                       assinitial=0.01,
                       assdecay=1000, # ass means actor steps size
                       rdecay=0.95, # reward decay weight
                       #  parambound=None # bound for the parameters
                       parambound=[[-50, 150], [-50, 50]]# bound for the parameters
                       )

agent = ActorCriticAgent(learner, sdim=8, adim=1)
experiment = Experiment(task, agent)

def loop():
    for i in xrange(ITER_NUM):
        reward = 0
        for j in xrange(LEARN_INTERVAL):
            reward += experiment._oneInteraction()
        agent.learn()

        # periodically reset stepsize to increase learning speed.
        if i % 1000 == 0:
            learner.resetStepSize()
        cPrint(iteration=i, th0=policy.theta[0], th1=policy.theta[1],
               reward=reward)
try:
    loop()
except KeyboardInterrupt:
    pass

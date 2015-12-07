#!/usr/bin/env python
__author__ = 'Jing Conan Wang, Boston University, wangjing@bu.edu'

from scipy import zeros

from collections import defaultdict

from pybrain.rl.experiments import Experiment
import librl
from librl.policies import BoltzmanPolicy
from librl.environments.mazes.trapmaze import TrapMaze
from librl.environments.mazes.tasks.robottask import RobotMotionAvgRewardTask
from librl.learners.td import TDLearner
from librl.agents.actorcriticagent import ActorCriticAgent
from librl.util import WriteTrace

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
task.GOAL_REWARD = 100
task.TRAP_REWARD = 0

policy = BoltzmanPolicy(4, T, iniTheta)
learner = TDLearner(policy, 0.9, 8, 1)
agent = ActorCriticAgent(learner, sdim=8, adim=1)
experiment = Experiment(task, agent)

#####################
#    Parameters     #
#####################
ITER_NUM = 1000000
#  LEARN_INTERVAL = 10
LEARN_INTERVAL = 1
OUTPUT_FILENAME = 'tdac.tr'

trace = defaultdict(list)
def loop():
    for i in xrange(ITER_NUM):
        reward = 0
        for j in xrange(LEARN_INTERVAL):
            reward += experiment._oneInteraction()
        agent.learn()

        # periodically reset stepsize to increase learning speed.
        if i % 1000 == 0:
            learner.resetStepSize()
        print 'theta value: [%f, %f], reward: %i '%(policy.theta[0],
                                                    policy.theta[1], reward)
        print 'theta value: [%f, %f]'%tuple(policy.theta)
        trace['iter'].append(i)
        trace['theta0'].append(policy.theta[0])
        trace['theta1'].append(policy.theta[1])
        trace['reward'].append(reward)

try:
    loop()
except KeyboardInterrupt:
    pass

WriteTrace(trace, OUTPUT_FILENAME)

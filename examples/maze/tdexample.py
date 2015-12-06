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
learner = TDLearner(policy, 0.9, 0.8, 1)
agent = ActorCriticAgent(learner, sdim=8, adim=1)
experiment = Experiment(task, agent)

#####################
#    Parameters     #
#####################
ITER_NUM = 100000
RECORD_INTERVAL = 10
OUTPUT_FILENAME = 'tdac.tr'

trace = defaultdict(list)
def loop():
    i = -1
    while i < ITER_NUM:
        i += 1
        reward = experiment._oneInteraction()
        agent.learn()
        if i % RECORD_INTERVAL == 0:
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

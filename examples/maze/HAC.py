#!/usr/bin/env python
__author__ = 'Jing Conan Wang, Boston University, wangjing@bu.edu'

import sys
sys.path.insert(0, "../../")
from scipy import zeros

from pybrain.rl.experiments import Experiment

from librl.policy import BoltzmanPolicy
from librl.environments.mazes.trapmaze import TrapMaze
from librl.environments.mazes.tasks import RobotMotionTask, SimpleTemporalLogic
from librl.learners import LSTDACLearner, TDLearner
from librl.learners import HACLearner
from librl.agents import ACAgent

# import global parameters
from problem_settings import gridSize, unsafeStates, iniState, goalStates, TP, DF, senRange
from problem_settings import T, iniTheta

# Create environment
# Add unsafe states
envMatrix = zeros(gridSize)
envMatrix[zip(*unsafeStates)] = -1
env = TrapMaze(envMatrix, iniState, goalStates, TP, DF)

# Create task
task = RobotMotionTask(env, senRange=senRange)
# task = SimpleTemporalLogic(env, senRange=senRange)

# policy = BoltzmanPolicy(feaDim = 2, numActions = 4, T = 2, iniTheta=[0, 0])
policy = BoltzmanPolicy(feaDim = 2, numActions = 4, T = T, iniTheta=iniTheta)
# learner = LSTDACLearner(lamb = 1, c = 0.8, D=2)
# learner = HACLearner(lamb = 0.9, c = 0.8, D=2)
learner = HACLearner(lamb = 0.9, c = 5, D=10)
# learner = HACLearner(lamb = 0.9, c = 1.2, D=4)
# learner = LSTDACLearner(lamb = 0.9, c = 0.8, D=2)
# learner = HACLearner(lamb = 0.9, c = 5, D=10)
# learner = HACLearner(lamb = 0.9, c = 2, D=10)
# learner = TDLearner(lamb = 1, c = 0.8, D=2)
# agent = ExplorerLearningAgent(policy, learner)
agent = ACAgent(policy, learner, sdim=8, adim=1)
experiment = Experiment(task, agent)

def main():
    i = -1
    # trace = dict(epsiode=[], rp=[])
    trace = dict(ep=[], reward=[], it=[],
            theta0=[], theta1=[])
    th_trace = dict(theta0=[], theta1=[], it=[])
    r = 0
    j = -1
    try:
        while True:
            j += 1
            reward = experiment._oneInteraction()
            r += reward
            agent.learn()
            # if j == 1e6: break
            if j == 1e5: break
            if j % 100 == 0:
                print 'theta value: [%f, %f]'%tuple(policy.theta)
                th_trace['theta0'].append(policy.theta[0])
                th_trace['theta1'].append(policy.theta[1])
                th_trace['it'].append(j)

            if task.reachGoalFlag:
                i += 1
                if i == 5e3: break
                trace['theta0'].append(policy.theta[0])
                trace['theta1'].append(policy.theta[1])
                trace['ep'].append(i)
                trace['reward'].append(r)
                trace['it'].append(j)
                print '[%i]reach goal, reward'%(i), r
                r = 0
                task.reset()
                learner.resetStepSize()
                continue
    except KeyboardInterrupt:
        pass
# except Exception as e:
        # pass

    from librl.util import WriteTrace
    WriteTrace(trace, 'hac.tr')
    WriteTrace(th_trace, 'hac_theta.tr')

if __name__ == "__main__":
    from time import clock
    start = clock()
    main()
    end = clock()
    print 'total time is %i'%(end-start)




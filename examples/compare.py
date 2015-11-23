#!/usr/bin/env python

from scipy import zeros, array
import pylab

from pybrain.rl.environments.mazes import Maze
from pybrain.rl.experiments import *
from task import *
from environments import *
from learners import *
from agents import *
from policy import *
from ReachProbCalculator import *

import cPickle as pickle
from time import clock as now

# import global parameters
import settings
from settings import gridSize, unsafeStates, goalStates, TP, iniState, VISUAL, DF, iniTheta, lamb, c, D, showInterval, T, senRange, uSize, hessianThetaTh

# Create environment
# Add unsafe states
envMatrix = zeros(gridSize)
envMatrix[zip(*unsafeStates)] = -1
sDim = 2
aDim = 1

# stepNum = 6000
stepNum = 20000

LSTDAC_TRACE_FILE = './res/lstdac.p'
HESSIANAC_TRACE_FILE = './res/hessian.p'
ENAC_TRACE_FILE = './res/enac.p'
COMP_FIG_FILE = './res/comp.eps'
COMP_FIG_FILE_REAL_T = './res/com-rt.eps'
LR = 0.3

def ConTest(experiment, task, agent, learner, reachProb, traceFile):
    trace = dict(rp=[], ts=[], rts=[])
    startT = now()
    for i in range(stepNum):
        experiment.doInteractions(1)
        if task.reachGoalFlag:
            agent.reset()
            task.reset()
            continue
        agent.learn()

        if i % settings.showInterval == 0: print 'theta value, [%f, %f]'%tuple(agent.learner.theta)
        if settings.CAL_EXACT_PROB and i % settings.reachProbInterval == 0:
            rp, tt = reachProb.GetReachProb(agent.learner.theta)
            print 'iter: [%d] reachProb: %f,  aveCost: %f it takes, %f seconds' %(i, rp, 1.0 / rp - 1, tt)
        trace['rp'].append(rp)
        trace['ts'].append(i)
        trace['rts'].append(now()-startT)
    # pickle.dump(trace, open('./res/lstdac.p', 'w'))
    pickle.dump(trace, open(traceFile, 'w'))


def LSTDAC_TEST():
    # Create Testbed for LSTDAC
    env = TrapMaze(envMatrix, iniState, goalStates, TP, DF)
    task = RobotMotionTask(env, senRange=senRange)
    policy = BoltzmanPolicy(T)
    learner = LSTDACLearner(aDim, iniTheta,
            lamb=lamb, c=c, D=D)
    agent = LSTDACAgent(policy, learner, sDim, aDim)
    experiment = Experiment(task, agent)
    reachProb = ReachProbCalculator(env, task, agent)
    ConTest(experiment, task, agent, learner, reachProb, LSTDAC_TRACE_FILE)



def HessianAC_TEST():
    env = TrapMaze(envMatrix, iniState, goalStates, TP, DF)
    task = RobotMotionTask(env, senRange=senRange)
    policy = BoltzmanPolicy(T)
    learner = HessianACLearner(aDim, iniTheta,
            uSize=uSize, gridSize=gridSize, lamb=lamb,
            c=c, D=D, hessianThetaTh=hessianThetaTh)
    agent = HessianACAgent(policy, learner, sDim, aDim)
    experiment = Experiment(task, agent)
    reachProb = ReachProbCalculator(env, task, agent)
    ConTest(experiment, task, agent, learner, reachProb, HESSIANAC_TRACE_FILE)


def ENACAC_TEST():
    # Create Testbed for LSTDAC
    env = TrapMaze(envMatrix, iniState, goalStates, TP, DF)
    task = RobotMotionTask(env, senRange=senRange)
    policy = BoltzmanPolicy(T)
    learner = ENAC(iniTheta=iniTheta, learningRate=LR)
    agent = LSTDACAgent(policy, learner, sDim, aDim)
    experiment = Experiment(task, agent)
    reachProb = ReachProbCalculator(env, task, agent)
    # epsiodNum = 10
    trace = dict(rp=[], ts=[], rts=[])
    t = 0
    # for i in xrange(epsiodNum):
    i = 0
    startT = now()
    while True:
        i += 1
        experiment = EpisodicExperiment(task, agent)
        all_rewards = experiment.doEpisodes(1)
        agent.learn()
        print 'theta value, ', learner.theta
        rp, tt = reachProb.GetReachProb(agent.learner.theta)
        print 'iter: [%d] reachProb: %f,  aveCost: %f it takes, %f seconds' %(i, rp, 1.0 / rp - 1, tt)
        trace['rp'].append(rp)
        t += len(all_rewards[0])
        trace['ts'].append(t)
        trace['rts'].append(now()-startT)
        if t > stepNum:
            break
    pickle.dump(trace, open(ENAC_TRACE_FILE, 'w'))

def Visualize():
    lstdacTrace = pickle.load(open(LSTDAC_TRACE_FILE, 'r'))
    hessianTrace = pickle.load(open(HESSIANAC_TRACE_FILE, 'r'))
    enacTrace = pickle.load(open(ENAC_TRACE_FILE, 'r'))
    mp = {'lstdac':lstdacTrace,
            'hessian':hessianTrace,
            'enac':enacTrace}
    pylab.figure()
    for k, tr in mp.iteritems():
        pylab.plot(tr['ts'], tr['rp'])
    pylab.legend(mp.keys())
    pylab.xlabel('step')
    pylab.ylabel('reachability probability')
    pylab.savefig(COMP_FIG_FILE)

    pylab.figure()
    for k, tr in mp.iteritems():
        pylab.plot(tr['rts'], tr['rp'])
    pylab.legend(mp.keys())
    pylab.xlabel('time (s)')
    pylab.ylabel('reachability probability')
    pylab.savefig(COMP_FIG_FILE_REAL_T)

    # pylab.show()
    # import pdb;pdb.set_trace()

def main():
    ENACAC_TEST()
    LSTDAC_TEST()
    HessianAC_TEST()
    Visualize()
    pass


if __name__ == "__main__":
    main()









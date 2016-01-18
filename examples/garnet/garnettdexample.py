
#!/usr/bin/env python
__author__ = 'Jing Conan Wang, Boston University, wangjing@bu.edu'

import scipy
from scipy import zeros

from collections import defaultdict

from pybrain.rl.experiments import Experiment
import librl
from librl.policies import *
from librl.environments.garnet import *
from librl.learners import *
from librl.agents.actorcriticagent import ActorCriticAgent
from librl.util import cPrint
from librl.experiments import *


ITER_NUM = 50000
LEARN_INTERVAL = 1

# Create environment
# Add unsafe states
feaDim = 8
numActions = 4
paramDim = numActions * feaDim
env = GarnetEnvironment(numStates=30,
                        numActions=numActions,
                        branching=2,
                        feaDim=feaDim,
                        feaSum=3,
                        savePath='./garnet_testbed.pkz')

# Create task
task = GarnetTask(env, sigma=0.1)
#  task = GarnetLookForwardTask(env, sigma=0.1)

bound = [(-100, 100)] * (numActions * feaDim)
policy = BoltzmanPolicy(4, T=1, theta=scipy.zeros((paramDim,)))
featureModule = PolicyValueFeatureModule(policy, 'bsglpolicywrapper')
#  learner = HessianLearner(hessianlearningrate=0.9, policy=policy,
#  LSTDLearner.enableOnlyEssentialFeatureInCritic = False
#  learner = LSTDLearner(policy=policy,
learner = TDLearner(module=featureModule,
                    cssinitial=0.1,
                    cssdecay=1000,
                    assinitial=0.01,
                    assdecay=1000, # ass means actor steps size
                    rdecay=0.95,
                    maxcriticnorm=10000, # maximum critic norm
                    tracestepsize=0.9 # stepsize of trace
                    )

agent = ActorCriticAgent(learner, sdim=paramDim*numActions, adim=1)
experiment = SessionExperiment(task, agent, policy=policy)

try:
    experiment.doSessionsAndPrint(sessionNumber=ITER_NUM, sessionSize=LEARN_INTERVAL)
except KeyboardInterrupt:
    pass

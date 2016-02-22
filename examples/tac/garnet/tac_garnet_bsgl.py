
#!/usr/bin/env python
__author__ = 'Jing Conan Wang, Boston University, wangjing@bu.edu'

import scipy
from scipy import zeros

from collections import defaultdict

from librl.agents.actorcriticagent import ActorCriticAgent
from librl.environments.garnet import *
from librl.experiments import *
from librl.learners import *
from librl.learners.bsgl import *
from librl.policies import *
from librl.util import cPrint
from pybrain.rl.experiments import Experiment

import garnetproblem as prob
import librl

####### parameters ##############
#  learnerClass = BSGLRegularGradientActorCriticLearner
learnerClass = BSGLFisherInfoActorCriticLearner
#  learnerClass = BSGLAdvParamActorCriticLearner
#  learnerClass = BSGLAdvParamFisherInfoActorCriticLearner


# Learner will learn after every session.
#  sessionNumber = 50000
#  sessionNumber = 1000000
sessionNumber = 1000
sessionSize = 1000

# the dimensionality of parameters in our policy is equal to the
# dimensionality of state feature vector.
# for garnet task, dimensionality is higher because feaDim is state feature,
# and action feature = feaDim * numActions
paramDim = prob.feaDim * prob.numActions

# Bound of parameters.
bound = [(-100, 100)] * paramDim

# initial critic stepsize (alpha_0 in the paper).
cssinitial = 0.1
# critic stepsize decay factor (alpha_c in the paper).
cssdecay = 1000

# initial actor stepsize (beta_0 in the paper).
assinitial = 0.01
# actor stepsize decay factor (beta_c in the paper).
assdecay = 1000

# According to BSGL paper, the stepsize for average reward is set as rdecay *
# critic_stepsize.
rdecay = 0.95

# temperature of boltzmann policy.
T = 1

# max norm of the critic parameter.
maxcriticnorm = 1000000

# initial value for parameters.
initialTheta = scipy.zeros((paramDim,))

# dimensionality of the observation
obsDim = paramDim * prob.numActions

#################################

policy = BoltzmanPolicy(prob.numActions, T=T, theta=initialTheta)
featureModule = PolicyValueFeatureModule(policy, 'bsglpolicywrapper')
learner = learnerClass(module=featureModule,
                      cssinitial=cssinitial,
                      cssdecay=cssdecay,
                      assinitial=assinitial,
                      assdecay=assdecay, # ass means actor steps size
                      rdecay=rdecay,
                      maxcriticnorm=maxcriticnorm, # maximum critic norm
                      parambound=bound
                      )

agent = ActorCriticAgent(learner, sdim=obsDim, adim=1, batch=True)
# bsgl method only takes GarnetTask as input in which state action feature is
# created by padding state feature
experiment = SessionExperiment(prob.bsgl_task, agent, policy=policy, batch=True)

try:
    experiment.doSessionsAndPrint(sessionNumber=sessionNumber,
                                  sessionSize=sessionSize)
except KeyboardInterrupt:
    pass

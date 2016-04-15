
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
import garnetproblem as prob

####### parameters ##############
# Learner will learn after every session.
#  sessionNumber = 50000
sessionNumber = 1000
sessionSize = 1000

# the dimensionality of parameters in our policy is equal to the
# dimensionality of state feature vector.
paramDim = prob.feaDim

# Bound of parameters.
bound = [(-50, 50)] * paramDim

# initial critic stepsize (alpha_0 in the paper).
cssinitial = 0.1
# critic stepsize decay factor (alpha_c in the paper).
cssdecay = 1000

# initial actor stepsize (beta_0 in the paper).
assinitial = 0.1
# actor stepsize decay factor (beta_c in the paper).
assdecay = 1000

# According to BSGL paper, the stepsize for average reward is set as rdecay *
# critic_stepsize.
rdecay = 0.95

# temperature of boltzmann policy.
T = 1

# max norm of the critic parameter.
maxcriticnorm = 100000

# trace stepsize (lambda in the paper).
tracestepsize = 0.7

# initial value for parameters.
initialTheta = scipy.zeros((paramDim,))

# dimensionality of the observation
obsDim = prob.feaDim * prob.numActions

#################################

policy = BoltzmanPolicy(prob.numActions, T=T, theta=initialTheta)
featureModule = PolicyFeatureModule(policy, 'bsglpolicywrapper')
learner = LSTDLearner(module=featureModule,
                      cssinitial=cssinitial,
                      cssdecay=cssdecay,
                      assinitial=assinitial,
                      assdecay=assdecay, # ass means actor steps size
                      rdecay=rdecay,
                      maxcriticnorm=maxcriticnorm, # maximum critic norm
                      tracestepsize=tracestepsize, # stepsize of trace
                      parambound=bound
                      )

agent = ActorCriticAgent(learner, sdim=obsDim, adim=1, batch=True)
experiment = SessionExperiment(prob.task, agent, policy=policy, batch=True)

try:
    experiment.doSessionsAndPrint(sessionNumber=sessionNumber,
                                  sessionSize=sessionSize)
except KeyboardInterrupt:
    pass

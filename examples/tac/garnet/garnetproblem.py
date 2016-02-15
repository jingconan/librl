from librl.environments.garnet import *

# Number of states (n in the paper).
numStates = 50
# Number of actions at each state (m in the paper).
numActions = 4
# Number of next states (b in the paper).
branching = 5
# Variance of the expected reward.
sigma = 0.1

# Dimension of the state feature (d in the paper).
feaDim = 8
# # of 1 in the state feature (l in the paper).
feaSum = 3
env = GarnetEnvironment(numStates=numStates,
                        numActions=numActions,
                        branching=branching,
                        feaDim=feaDim,
                        feaSum=feaSum,
                        loadPath='tac_results/garnet/garnet_testbed.pkz')


task = GarnetLookForwardTask(env, sigma=sigma)

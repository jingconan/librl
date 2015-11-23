# Gloabal Parameters

#############################
###      Scene Setting    ###
#############################
iniState = (0, 0)
# goalStates = [ (49, 0), (49, 49), (0, 49) ]
goalStates = [ (49, 49) ]
# goalStates = [ (49, 0), (49, 49), (0, 49) ]
gridSize = [50, 50]

unsafeRegion = [
        # (10, 0, 1, 10),
        # (10, 15, 1, 25),
        # (11, 15, 20, 1),
        # (10, 10, 20, 1),
        # (0, 20, 5, 1),
        # (0, 30, 10, 1),
        # (0, 40, 5, 1),

        (15, 0, 9, 6),
        (42, 0, 3, 3),
        (29, 5, 3, 3),
        (5, 7, 6, 3),
        (37, 7, 6, 6),
        (23, 11, 6, 6),
        (0, 15, 3, 3),
        (6, 20, 3, 6),
        (18, 22, 12, 3),
        (44, 20, 6, 3),
        (11, 30, 6, 9),
        (23, 34, 3, 3),
        (32, 33, 9, 9),
        (47, 39, 3, 6),
        (0, 38, 6, 6),
        # (17, 47, 9, 3)
        ]

def IsSafe(st):
    x, y = st
    for region in unsafeRegion:
        (xCoord, yCoord, xLen, yLen) = region
        if ( x - xCoord >= 0 ) and ( x - xCoord < xLen ) \
        and ( y - yCoord >= 0) and ( y - yCoord < yLen ):
            return 0
    return 1

unsafeStates = []
for x in xrange(gridSize[0]):
    for y in xrange(gridSize[1]):
        if not IsSafe([x, y]):
            unsafeStates.append([x, y])

#############################
## initial value for theta ##
#############################
# iniTheta = [0, 0.5]
# iniTheta = [-0.5, 0.5]
# iniTheta = [-1, 1]
# iniTheta = [0, 0]
# iniTheta = [6, -17]
# iniTheta = [6, -10]
# iniTheta = [5, -1]
# iniTheta = [0, 0]
# iniTheta = [50, -10]
# iniTheta = [0, -10]
# iniTheta = [10, -20]
# iniTheta = [0, -10]
# iniTheta = [0, 0]
# iniTheta = [100, 100]
# iniTheta = [10, 10]
# iniTheta = [5, 5]
# iniTheta = [67, 36]
# iniTheta = [100, 100]
# iniTheta = (10, 10)
iniTheta = [10, 10]
# iniTheta = [0, 0]
# iniTheta = [3, 3]
# iniTheta = [10, 1]
# iniTheta = [0, 0]
# iniTheta = [100, 1]
# iniTheta = [10, 1]


n = len(iniTheta) # dimension of the theta
uSize = 4  # Control Size
stateSize = gridSize[0] * gridSize[1] # state size

# Threshold For Algorithm Convergence
# th = 1e-4
# minimial iteration number
# minIter = 3e4
# minIter = 4e4


# for test
th = 1e-2
minIter = 8e2

# TP is transition probability
# The order is ENWS
# TP = [[0.7, 0.1, 0.1, 0.1],
#         [0.1, 0.7, 0.1, 0.1],
#         [0.1, 0.1, 0.7, 0.1],
#         [0.1, 0.1, 0.1, 0.7]]
# TP = [[1, 0, 0, 0],
#         [0, 1, 0, 0],
#         [0, 0, 0, 1],
#         [0, 0, 0, 1]]

# TP = [[0.85, 0.05, 0.05, 0.05],
        # [0.05, 0.85, 0.05, 0.05],
        # [0.05, 0.05, 0.85, 0.05],
        # [0.05, 0.05, 0.05, 0.85]]

TP = [[0.97, 0.01, 0.01, 0.01],
        [0.01, 0.97, 0.01, 0.01],
        [0.01, 0.01, 0.97, 0.01],
        [0.01, 0.01, 0.01, 0.97]]

# showInterval = 100
showInterval = 1000 # interval between two consequent show of theta value
# showInterval = 1

# Calculate L1 Distance
# DF = lambda x, y:abs(x[0] - y[0]) + abs(x[1] - y[1])
from math import sqrt
# DF = lambda x, y:sqrt( (x[0] - y[0])**2 + (x[1] - y[1])**2 )
DF = lambda x, y: abs(x[0] - y[0]) + abs(x[1] - y[1])

#####################################################
#------ Setting For ReachProbCalculator.py -------- #
#####################################################
# CAL_EXACT_PROB = False
CAL_EXACT_PROB = True
reachProbInterval = 100

## Calculate the value surface ##

#####################################################
#------ Setting For Hesssian AC     -------- #
#####################################################
# hessianThetaTh = 2
# hessianThetaTh = 0.01
hessianThetaTh = 0.1 # threshold to detemin whether to accept this hessian based update



######## Step size parameter #########
lamb=0.9
# lamb = 0.95
# c=0.05
# c = 0.5
# c = 0.05
# D=5
c = 5
D = 10

# the neighbor of state is states where distance with x is <= senRange.
senRange = 1
# senRange = 2
# senRange = 3
#####################################################
#------ Setting For Display  -------- #
#####################################################
# VISUAL = True
VISUAL = False


# allivate the poor structure of the policy
ratioTh = 4


##########################
####          MISC      ##
##########################

CALW_OPTION = 'EXP'

# ACTION_OPTION = 'ALWAYS'
# ACTION_OPTION = 'NULL'
# CALW_OPTION = 'TANH'


from numpy import exp
def CalW_EXP(score, theta):
    perfer = sum( s * t for s, t in zip(score, theta) )
    return exp(perfer/T)
    # return min([ np.exp(perfer/T), 1e4])

from numpy import tanh
def CalW_TANH(score, theta):
    perfer = sum( s * t for s, t in zip(score, theta) )
    return 1/8.0 * ( tanh(perfer / T) + 1 )
CalWDict = {'EXP':CalW_EXP, 'TANH':CalW_TANH}
CalW = CalWDict[CALW_OPTION]


ROOT = '.'
THETA_FILE = ROOT + '/res/theta.txt'
RP_FILE = ROOT + '/res/rp.txt'
# AC_TYPE = 'hessian'
AC_TYPE = 'normal'




# T = 20
T = 7
# T = 10
# T = 5

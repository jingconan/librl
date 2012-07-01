from problem_settings import gridSize, unsafeStates, iniState, goalStates, TP, DF, senRange
# import pylab, time
from numpy import zeros
from matplotlib.pyplot import *

envMatrix = zeros(gridSize)
envMatrix[zip(*unsafeStates)] = -1
# env = TrapMaze(envMatrix, iniState, goalStates, TP, DF)

gray()
ion()

cFlag = dict(gs=-2, robot=2, trap=-1, normal=0)
visEnvMat = envMatrix
for gs in goalStates:
    visEnvMat[tuple(gs)] = cFlag['gs']

pcolor(visEnvMat)
grid(True)
draw()
savefig('scene.eps')
show()



#!/usr/bin/env python
from HAC import *
import sys
sys.path.insert(0, '..')
import settings
from ReachProbCalculator import *

reachProb = ReachProbCalculator(env, task, agent)

def PlotCostSurface():

    xRange = settings.xRange
    yRange = settings.yRange
    import numpy as np
    totalNum = len(xRange) * len(yRange)
    yLen = len(yRange)
    reachProbMat = np.zeros( (len(xRange), yLen) )
    i = 0
    for x in xRange:
        j = 0
        for y in yRange:
            rp, time = reachProb.GetReachProb([x, y])
            reachProbMat[i, j] = rp
            seq = i*yLen+j
            if seq % settings.ProgressShowInterval == 0:
                print '[%f %%] x: %f, y: %f ' %( seq * 100.0/totalNum, x, y)
                print 'expected totalTime is: %d s' %( totalNum * time )
            j += 1
        i += 1


    import matplotlib.pyplot as plt
    from matplotlib import cm
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)
    X, Y = np.meshgrid(yRange, xRange)
    surf = ax.plot_surface(X, Y, reachProbMat, rstride=1, cstride=1, cmap=cm.jet)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # fig.savefig('./res/reachProb.eps')
    plt.title('surface of reachability probability')
    plt.ylabel('safety weight')
    plt.xlabel('progress weight')
    fig.savefig('reachProb.eps')

if __name__ == "__main__":
    PlotCostSurface()



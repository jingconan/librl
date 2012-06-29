#!/usr/bin/env python
from HAC import *
import sys
sys.path.insert(0, '..')
from ReachProbCalculator import *

reachProb = ReachProbCalculator(env, task, agent)

def loadtxt(fname):
    fid = open(fname)
    value = []
    while True:
        line = fid.readline()
        if not line: break
        if line[0] == '#':
            kl = line.rsplit(' ')
            keys = [k.strip('#\n') for k in kl]
            continue
        val = line.rsplit(' ')
        value.append([float(v) for v in val])
    return keys, value

def loadTheta(fname):
    # fid = open('./res/lstdac_theta.tr')
    fid = open(fname)
    theta = []
    while True:
        line = fid.readline()
        if not line: break
        if line[0] == '#': continue
        val = line.rsplit(' ')
        theta.append([float(v) for v in val])
    return theta

from matplotlib.pyplot import *
def plotRP(fname, ptNum, showFlag=True):
    keys, values = loadtxt(fname)
    col_values = zip(*values)
    theta0 = col_values[keys.index('theta0')]
    theta1 = col_values[keys.index('theta1')]
    it = col_values[keys.index('it')]

    n = len(theta0)
    stepSize = n / ptNum if ptNum else 1
    trace = dict(rp=[], time=[], it=[])
    for i in xrange(0, n, stepSize):
        rp, time = reachProb.GetReachProb([theta0[i], theta1[i]])
        trace['rp'].append(rp)
        trace['time'].append(time)
        trace['it'].append(it[i])
    plot(trace['it'], trace['rp'])
    if showFlag: show()

if __name__ == "__main__":
    # plotRP('./enac.tr', 50)
    # plotRP('./hac.tr', 50, False)
    # plotRP('./lstdac.tr', 50, False)
    # plotRP('./tdac.tr', 50, False)

    plotRP('./hac.tr', 200, False)
    plotRP('./lstdac.tr', 200, False)
    plotRP('./tdac.tr', 200, False)
    # show()
    # plotRP('./res/lstdac_theta.tr', 50, False)
    # plotRP('./res/tdac_theta.tr', 50, False)
    # plotRP('./res/enac_theta.tr', 50, False)
    # legend(['lstdac', 'tdac'])
    legend(['hac','lstdac', 'tdac'])
    show()

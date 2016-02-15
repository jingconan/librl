#!/usr/bin/env python
"""A utility program to plot the reward with confidence interval.

It takes output of the multirun.py and plot the reward mean with 95%
confidence interval.

Sample Command:
./blaze run ./tools/analyzetrace.py ./sample_results/lstd_new_test@50
"""
import fileinput
import numpy as np
import pylab as P
import pandas
import argparse
import scipy.stats as ss
import librl
from collections import defaultdict
from librl.util import createShardFilenames

#########################
# Parameters
#  windowSize = 5000
windowSize = 1
sampleInterval = 1 # sample the raw data
#  plotInterval = 1000 # limit # of pts in the output
plotInterval = 1 # limit # of pts in the output
#########################

def parseLine(line):
    tokens = line.split(',')
    result = []
    for t in tokens:
        subtokens = t.split(':')
        result.append((subtokens[0], float(subtokens[1])))
    return result

def addRecord(data, record):
    for k, v in record:
        data[k].append(v)

def loadTrace(filename, fields=None):
    print 'load file: ', filename
    data = defaultdict(list)
    i = 0
    with open(filename, 'r') as fObj:
        for line in fObj:
            if not line or line[0] == '-':
                continue
            i += 1
            if i == sampleInterval:
                i = 0
                # parse the line and extract values
                tokens = line.split(',')
                for t in tokens:
                    subtokens = t.split(':')
                    if fields is not None and subtokens[0] in fields:
                        data[subtokens[0]].append(float(subtokens[1]))
                        break

                    data[subtokens[0]].append(float(subtokens[1]))
    return data

def rollingMean(data, windowSize, interval):
    N = len(data)
    result = []
    for i in xrange(0, N, interval):
        result.append(np.mean(data[i:i+windowSize]))
    return result

def multipleRunRollingMean(mData, field, windowSize, interval):
    minLen = min([len(data[field]) for data in mData])
    windowStats = defaultdict(list)
    for i in xrange(0, minLen, interval):
        windowData = []
        for data in mData:
            windowData.extend(data[field][i:i+windowSize])
        windowStats['mean'].append(np.mean(windowData))
        windowStats['std'].append(np.std(windowData))
    return windowStats

def plotWithCI(x, y, yerr):
    P.plot(x, y, '-')
    P.plot(x, y+yerr, '--g')
    P.plot(x, y-yerr, '--g')

parser = argparse.ArgumentParser(description='Analyze run output')
parser.add_argument('filename', help='the path of trace files')
args = parser.parse_args()

if '@' in args.filename:
    tokens = args.filename.split('@')
    assert len(tokens) == 2, 'wrong format of shared file'
    filenames = createShardFilenames(tokens[0], int(tokens[1]))
else:
    filenames = [args.filename]

mData = [loadTrace(filename, ('reward')) for filename in filenames]
windowStats = multipleRunRollingMean(mData, 'reward', windowSize, plotInterval)
ptNumber = len(windowStats['mean'])

runNumber = len(filenames)
rewardMean = windowStats['mean']
rewardStd= windowStats['std']
sampleNumber = runNumber * windowSize
rewardSem = rewardStd / (P.sqrt(sampleNumber))
confidenceLevel = 0.95
yerr = ss.t.ppf((confidenceLevel + 1) / 2.0, sampleNumber-1) * rewardSem
plotWithCI(range(ptNumber), rewardMean, yerr)
#  P.errorbar(range(ptNumber), rewardMean, yerr=yerr, fmt='--o', ecolor='g',
#             capthick=2)
P.show()

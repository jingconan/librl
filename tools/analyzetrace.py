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
from librl.util import createShardFilenames

#########################
# Parameters
windowSize = 10
sampleInterval = 10
#########################

def parseLine(line):
    tokens = line.split(',')
    result = {}
    for t in tokens:
        subtokens = t.split(':')
        result[subtokens[0]] = float(subtokens[1])
    return result

def loadTrace(filename):
    data = []
    for line in open(filename, 'r'):
        if line.startswith('-->'):
            continue
        data.append(parseLine(line))

    return pandas.DataFrame(data)

parser = argparse.ArgumentParser(description='Analyze run output')
parser.add_argument('filename', help='the path of trace files')
args = parser.parse_args()

if '@' in args.filename:
    tokens = args.filename.split('@')
    assert len(tokens) == 2, 'wrong format of shared file'
    filenames = createShardFilenames(tokens[0], int(tokens[1]))
else:
    filenames = [args.filename]

# Analyze data
multiRunData = []
minLen = float('inf')
for filename in filenames:
    data = loadTrace(filename)
    rm = pandas.rolling_mean(data.reward, windowSize) #mean of reward
    if len(rm) < minLen:
        minLen = len(rm)
    multiRunData.append(rm)

fileNumber = len(filenames)
ptNumber = int(minLen / sampleInterval)
print('ptNumber', ptNumber)
combinedResult = P.zeros((ptNumber, fileNumber))
for i in xrange(fileNumber):
    combinedResult[:, i] = multiRunData[i][0:minLen:sampleInterval]

rewardMean = P.mean(combinedResult, axis=1)
rewardStdDev = P.std(combinedResult, axis=1)
yerr = ss.t.ppf(0.95, fileNumber) * rewardStdDev
P.errorbar(range(ptNumber), rewardMean, yerr=yerr, fmt='--o', ecolor='g',
           capthick=2)
P.show()

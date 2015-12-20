import fileinput
import numpy as np
import pylab as P
import pandas

def parseLine(line):
    tokens = line.split(',')
    result = {}
    for t in tokens:
        subtokens = t.split(':')
        result[subtokens[0]] = float(subtokens[1])
    return result

def loadTrace():
    data = []
    for line in fileinput.input():
        if line.startswith('-->'):
            continue
        data.append(parseLine(line))

    return pandas.DataFrame(data)

pdata = loadTrace()
windowsize = 1000
reward_mean = pandas.rolling_mean(pdata.reward, windowsize)
P.plot(reward_mean)
P.xlabel('iter')
P.ylabel('reward moving average')
P.show()

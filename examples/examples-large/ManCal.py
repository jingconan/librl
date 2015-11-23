#!/usr/bin/env python
"""
check reachability probability manually
"""
from HAC import *
from ReachProbCalculator import *
reachProb = ReachProbCalculator(env, task, agent)

while True:
    string = raw_input('please input the theta: ')
    theta = [float(v) for v in string.rsplit(',')]
    rp, time = reachProb.GetReachProb(theta)
    print 'rp, ', rp
    print 'time, ', time


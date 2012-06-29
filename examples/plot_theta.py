#!/usr/bin/env python
from pylab import *
x = loadtxt('lstdac_theta.tr')
# x = loadtxt('hac_theta.tr')
y = zip(*x)
plot(y[1])
show()

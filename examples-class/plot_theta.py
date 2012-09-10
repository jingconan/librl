#!/usr/bin/env python
from pylab import *
import numpy as np
def plot_theta(name, interval, marker):
    x = loadtxt(name)
    y = zip(*x)
    n = len(y[0])
    z = np.diff(y)
    print 'z, ', z
    theta_diff_mod = np.sqrt(z[0, :] ** 2 + z[1, :] **2)
    # interval = 1000
    theta_diff_mean= []
    for v in xrange(interval, n):
        theta_diff_mean.append(np.mean(theta_diff_mod[(v-interval):v]))
    plot(theta_diff_mean, marker)

if __name__ == "__main__":
    plot_theta('./res/lstdac_theta.tr', 1000, '-')
    plot_theta('./res/hac_theta.tr', 1000, '-.')
    # plot_theta('./res/nac_theta.tr', 1)
    legend(['lstdac', 'hac'])
    # legend(['lstdac', 'hac', 'nac'])
    savefig('./theta_diff_comp.eps')
    show()

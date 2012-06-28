#!/usr/bin/env python

from scipy import cumsum, divide, array
from matplotlib.pyplot import *
# FILE = "./rec_Q_Lin_5000.tr"
def plot_reward(FILE):
    lines = [ [float(v) for v in line.rsplit(' ')] for line in open(FILE) if line[0] is not '#']
    n = len(lines)
    trace = zip(*lines)
    cum_mean = divide( cumsum(trace[1]),
            1+ array(range(n)) )
    head = 30
    plot(trace[0][head:], cum_mean[head:])
    show()
    grid()
figure()
# FILE = "./QLambda_LinFA_rec.tr"
# plot_reward("./rec_Q_Lin_5000.tr")
# plot_reward("./QLambda_LinFA_rec.tr")
plot_reward("./lstdac.tr")
# plot_reward("./SARSA_TL.py")
legend(['Q_line', 'QLambda'])

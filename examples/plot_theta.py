from pylab import *
x = loadtxt('lstdac_theta.tr')
y = zip(*x)
plot(y[1])
show()

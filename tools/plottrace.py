#!/usr/bin/env python
import json
import pylab as P
import sys

methods = []
for i, filename in enumerate(sys.argv[1:]):
    methods.append(filename)
    data = json.load(open(filename, 'r'))
    P.plot(data['x'], data['y'], '-')

P.legend(methods, loc=4, fontsize=8)
#  P.xlim([0, 500])
P.show()

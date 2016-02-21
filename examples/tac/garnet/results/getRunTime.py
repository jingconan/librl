#!/usr/bin/env python
import sys
import re
import pylab as P

times = []
for i in xrange(1, len(sys.argv)):
    filename = sys.argv[i]
    with open(filename, 'r') as fid:
        content = fid.read()
        time_search = re.search('.*elapsed time: ([\d]+) seconds\n\n', content)
        if time_search:
            time = float(time_search.group(1))
            times.append(time)
print 'mean running time: ', P.mean(times)

#!/usr/bin/env python
import json
filenames = [
    './bsgl_alg1_1e6/fig.json',
    './bsgl_alg2_1e6/fig.json',
    './bsgl_alg3_1e6/fig.json',
    './bsgl_alg4_1e6/fig.json',
    './td_session_size_1000_num_1000/fig.json',
    './hessian0_fig.json',
    './hessian2_fig.json',
]
methods = [
    'BSGL-1',
    'BSGL-2',
    'BSGL-3',
    'BSGL-4',
    'Konda-TD',
    'Hessian',
    'Hessian2',
]
method_of_session_reward = ['Konda-TD', 'Hessian', 'Hessian2']
import pylab as P

for method, filename in zip(methods, filenames):
    data = json.load(open(filename, 'r'))
    if method in method_of_session_reward:
        data['y'] = P.array(data['y']) / 1000
    P.plot(data['x'], data['y'])

P.legend(methods, loc=4, fontsize=8)
P.show()

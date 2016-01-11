def CheckMember(obj, mSet):
    '''check whehter class .obj has member mSet'''
    if all( [ m in obj.__dict__.keys() for m in mSet ]):
        raise NotImplementedError('Not implement function')

import random
def GenRand(dist, select=False):
    '''Generate Random Variable According to Certain Kind of
    Distribution'''

    # TODO Finish A Complete Version
    s = 0
    rv = random.random()
    m = -1
    for p in dist:
        m += 1
        s += p
        if s > rv:
            break

    # print 'select', select
    # print 'val ', select == False
    if select == False:
        return m
    else:
        return select[m]
    # res = (select == False) and m or select[m]
    # return res

# For assert
import types

import inspect
VIS = False
# VIS = True
def debug(fn):
    def wrapped(*args, **kwargs):
        if VIS: print 'call: %s() in "%s"' %( fn.__name__, inspect.getsourcefile(fn).rsplit('/')[-1])
        return fn(*args, **kwargs)
    return wrapped

def func():
    print "hello world"

if __name__ == "__main__":
    func()


Expect = lambda X, P: sum( x * p for x, p in zip(X, P) )

import math
def WriteTrace(trace, fname):
    fid = open(fname, 'w')
    fid.write('#' + ' '.join(trace.keys()) + '\n')
    res = zip(*trace.values())
    for r in res:
        fid.write(' '.join([str(val) for val in r]) + '\n')

def cPrint(**kwargs):
    items = sorted(kwargs.items())
    print ','.join(['%s:%s' % (k, v) for k, v in items])


# def dotproduct(v1, v2):
    # return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    # return math.sqrt(dotproduct(v, v))
    return math.sqrt(dot(v.reshape(-1), v.reshape(-1)))

from numpy import dot
def angle(v1, v2):
    # return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2))) if length(v1) * length(v2) != 0 else None
    return math.acos(dot(v1.reshape(-1), v2.reshape(-1)) / (length(v1) * length(v2))) if length(v1) * length(v2) != 0 else None

def createShardFilenames(prefix, shardnumber):
    outputFilenames = []
    suffixLength = len(str(shardnumber))
    formatString = '%s_%0' + str(suffixLength) + 'd_of_' + str(shardnumber)
    for i in xrange(shardnumber):
        outputFilenames.append(formatString % (prefix, i))
    return outputFilenames


###############################
# Data Storage and Load
###############################
try:
    import cPickle as pickle
except ImportError:
    import pickle
import gzip
proto = pickle.HIGHEST_PROTOCOL
def zdump(obj, f_name):
    f = gzip.open(f_name,'wb', proto)
    pickle.dump(obj,f)
    f.close()

def zload(f_name):
    f = gzip.open(f_name,'rb', proto)
    obj = pickle.load(f)
    f.close()
    return obj


## Feature encoding and decoding.
import scipy
def encodeTriuAs1DArray(mat):
    """Encode the upper-triangular matrix as 1D array."""
    assert mat.shape[0] == mat.shape[1], 'only square matrix is supported!'
    iu = scipy.triu_indices(n=mat.shape[0], k=0)
    return mat[iu]

def decode1DArrayAsTriu(arr, n):
    "Decode 1D array as upper-triangular matrix"
    expectedLength = (n * n - n) / 2 + n
    assert expectedLength == len(arr), 'invalid input array.'
    iu = scipy.triu_indices(n)
    res = scipy.zeros((n, n))
    res[iu] = arr
    return res

def decode1DArrayAsSymMat(arr, n):
    res = decode1DArrayAsTriu(arr, n)
    return res + res.T - scipy.diag(scipy.diag(res))

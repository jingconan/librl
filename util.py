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

def WriteTrace(trace, fname):
    fid = open(fname, 'w')
    fid.write('#' + ' '.join(trace.keys()) + '\n')
    res = zip(*trace.values())
    for r in res:
        fid.write(' '.join([str(val) for val in r]) + '\n')




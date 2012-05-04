#!/usr/bin/env python
import os
runNum = 20
from shutil import copyfile as cp
LSTDAC_TRACE_FILE = './res/lstdac.p'
HESSIANAC_TRACE_FILE = './res/hessian.p'
ENAC_TRACE_FILE = './res/enac.p'
COMP_FIG_FILE = './res/comp.eps'
backFileList = [LSTDAC_TRACE_FILE, HESSIANAC_TRACE_FILE,
        ENAC_TRACE_FILE, COMP_FIG_FILE]

backFolder = './res/back/'
for i in xrange(runNum):
    os.system('./compare.py')
    for f in backFileList:
        nf = str(i) + '-' + f.rsplit('/')[-1]
        cp(f, backFolder + nf)



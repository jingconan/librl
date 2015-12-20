#!/usr/bin/env python
""" A utility program to run a command multiple times and save std output.

It save the std output to shareded file where each file has the format of
<prefix>_<shard_number>_of_<total_shard_number>. Each run is done by a
separate process, and a done file will be written when the process finishes.

Here is a handy shell function to check the progress. To use it, please add it
to your ~/.bashrc and run source ~/.bashrc in shell command

function check_process() {
  echo `ls $1 | grep ".done" | wc -l` out of \
       `ls $1 | grep -v ".done" | wc -l` runs has finished
}

Sample Commands: (in top folder)
./blaze run ./tools/multirun.py examples/maze/lstdexample.py
./sample_results/lstd_new_test@100
"""
import multiprocessing
from subprocess import check_call
from librl.util import createShardFilenames

import argparse
parser = argparse.ArgumentParser(description='run script multiple times')
parser.add_argument('script_filename',
                    help='path of the script that will be run')
parser.add_argument('output_filename',
                    help='output sharded filename. It has format of prefix@N, '
                         'where N is the # of runs')

args = parser.parse_args()

def myExecute(filename):
    check_call('./blaze run %s > %s' % (args.script_filename, filename),
               shell=True)
    check_call('touch %s.done' % (filename), shell=True)

tokens = args.output_filename.split('@')
assert len(tokens) == 2, 'wrong format of output file'
outputPrefix = tokens[0]
shardNumber = int(tokens[1])

p = multiprocessing.Pool()
outputFilenames = createShardFilenames(outputPrefix, shardNumber)
p.map(myExecute, outputFilenames)

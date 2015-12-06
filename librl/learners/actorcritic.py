from __future__ import print_function, division, absolute_import
from ..util import *
from pybrain.utilities import *

class ActorCriticLearner(object):
    """This is the basis class for all actor-critic method"""
    def learnOnDataSet(self):
        abstractMethod()


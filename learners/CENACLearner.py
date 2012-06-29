__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'


from pybrain.rl.learners.directsearch.enac import *
from pybrain.auxiliary import GradientDescent
from util import *


from scipy import ones, dot, ravel, array
from scipy.linalg import pinv



class CENAC(ENAC):
    """customized natural actor critic method"""
    @property
    def theta(self):
        return self.network.params

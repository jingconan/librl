__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'


from pybrain.rl.learners.directsearch.enac import *
from pybrain.auxiliary import GradientDescent
from util import *


from scipy import ones, dot, ravel, array
from scipy.linalg import pinv
import math

class CENAC(ENAC):
    """customized natural actor critic method"""
    def __init__(self):
        super(CENAC, self).__init__()
        self.last_gradient = None

    @property
    def theta(self):
        return self.network.params

    def learn(self):
        """ calls the gradient calculation function and executes a step in direction
            of the gradient, scaled with a small learning rate alpha. """
        assert self.dataset != None
        assert self.module != None

        # calculate the gradient with the specific function from subclass
        gradient = self.calculateGradient()
        # print 'gradient, ', gradient
        # gradient = array([[0.1], [0.1]])

        # scale gradient if it has too large values
        # if max(gradient) > 1000:
            # gradient = gradient / max(gradient) * 1000

        if max(gradient) > 5:
            gradient = gradient / max(gradient) * 5

        # if max(gradient) > 5:
            # gradient = gradient / max(gradient) * 5

        if self.last_gradient is None:
            self.last_gradient = gradient
            return
        print 'angle, ', angle(self.last_gradient, gradient)
        if angle(self.last_gradient, gradient) > 0.8:
            return

        # update the parameters of the module
        p = self.gd(gradient.flatten())
        self.network._setParameters(p)
        self.network.reset()

    def calculateGradient(self):
        # normalize rewards
        # self.dataset.data['reward'] /= max(ravel(abs(self.dataset.data['reward'])))

        # initialize variables
        R = ones((self.dataset.getNumSequences(), 1), float)
        X = ones((self.dataset.getNumSequences(), self.loglh.getDimension('loglh') + 1), float)

        # collect sufficient statistics
        # print 'seq, ', self.dataset.getNumSequences()
        for n in range(self.dataset.getNumSequences()):
            _state, _action, reward = self.dataset.getSequence(n)
            seqidx = ravel(self.dataset['sequence_index'])
            if n == self.dataset.getNumSequences() - 1:
                # last sequence until end of dataset
                loglh = self.loglh['loglh'][seqidx[n]:, :]
            else:
                loglh = self.loglh['loglh'][seqidx[n]:seqidx[n + 1], :]

            X[n, :-1] = sum(loglh, 0)
            R[n, 0] = sum(reward, 0)
        # import pdb;pdb.set_trace()

        # linear regression
        beta = dot(pinv(X), R)
        return beta[:-1]

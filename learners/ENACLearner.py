__author__ = 'Thomas Rueckstiess, ruecksti@in.tum.de'


from pybrain.rl.learners.directsearch.policygradient import *
from ActorCriticLearner import *
from pybrain.auxiliary import GradientDescent
from util import *


from scipy import ones, dot, ravel, array
from scipy.linalg import pinv



class ENAC(ActorCriticLearner):
    """ Episodic Natural Actor-Critic. See J. Peters "Natural Actor-Critic", 2005.
        Estimates natural gradient with regression of log likelihoods to rewards.
    """
    def __init__(self, **kwargs):
        iniTheta = kwargs['iniTheta']
        # self.gd = GradientDescent()
        # self.gd.init(array(iniTheta))
        self.theta = array( iniTheta )
        # self.gd.alpha = kwargs['learningRate']
        self.learningRate = kwargs['learningRate']
        self.feadim = len(iniTheta)
        self.loglh = LoglhDataSet(self.feadim)



    def calculateGradient(self):
        print 'calculateGradient'
        # normalize rewards
        # self.dataset.data['reward'] /= max(ravel(abs(self.dataset.data['reward'])))

        # initialize variables
        R = ones((self.dataset.getNumSequences(), 1), float)
        X = ones((self.dataset.getNumSequences(), self.loglh.getDimension('loglh') + 1), float)

        # collect sufficient statistics
        # print self.dataset.getNumSequences()
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

        # linear regression
        beta = dot(pinv(X), R)
        return beta[:-1]

    @debug
    def learnOnDataSet(self, dataset):
        self.dataset = dataset
        gradient = self.calculateGradient()

        # scale gradient if it has too large values
        if max(gradient) > 1000:
            gradient = gradient / max(gradient) * 1000

        # update the parameters of the module
        # self.theta = self.gd(gradient.flatten())
        self.theta = self.theta - self.learningRate * gradient.reshape(-1)

    def newEpisode(self):
        self.loglh.clear()


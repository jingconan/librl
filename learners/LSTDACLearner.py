__author__ = 'Jing Conan Wang, Boston University, wangjing@bu.edu'

# from pybrain.rl.learners.directsearch.policygradient import PolicyGradientLearner
from pybrain.rl.learners.directsearch.policygradient import *
from pybrain.utilities import *
from ActorCriticLearner import *
from util import *
import numpy as np
from scipy import ones, dot, ravel, zeros, array
from scipy.linalg import pinv

# class LSTDACLearner(PolicyGradientLearner):
class LSTDACLearner(ActorCriticLearner):
    ''' LSTD-Actor Critic Method. See `Least Squares Temporal Difference Actor-Critic Methods with Applications to Robot Motion Control <http://arxiv.org/abs/1108.4698>`_ for more information.
    Learner will learn from a data set. When the experiment is continous.
    the data set will only contains one point, and there will be some
    eligibility trace. Now we only consider the episodic case.
    '''
    def __init__(self, actiondim, iniTheta, **kwargs):
        # PolicyGradientLearner.__init__(self)
        self.feadim = len(iniTheta)
        self.loglh = LoglhDataSet(self.feadim)
        self.theta = array(iniTheta).reshape(-1, 1)

        self.k = 0
        self.z = np.zeros( (self.feadim, 1) )
        self.b = np.zeros( (self.feadim, 1) )
        self.r = np.zeros( (self.feadim, 1) )
        self.AE = np.zeros( (self.feadim, self.feadim) )
        self.alpha = 0;
        self.lastobs = None

        self.lamb = kwargs['lamb']
        self.c = kwargs['c']
        self.D = kwargs['D']

    def newEpisode(self):
        self.k = 0
        n = self.feadim
        self.z = np.zeros( (n, 1) )
        self.b = np.zeros( (n, 1) )
        self.r = np.zeros( (n, 1) )
        self.AE = np.zeros( (n, n) )
        self.alpha = 0
        self.lastobs = None


    def setReachProbCal(self, reachProbCal):
        self.reachProbCal = reachProbCal

    def Critic(self, xk, uk, gk, xkp1, ukp1, xkPsi, xkp1Psi):
        # ----- Critic --------
        z, b, r, AE, lamb, alpha, k = self.z, self.b, self.r, self.AE, self.lamb, self.alpha, self.k
        # print 'k, ', k
        xkPsi = xkPsi.reshape(-1, 1)
        xkp1Psi = xkp1Psi.reshape(-1, 1)
        z = lamb * z + xkPsi
        gam = 1.0 / (k+1)
        alpha += gam * (gk - alpha)

        b += gam * ( z * ( gk - alpha) - b )
        psiTmp = xkp1Psi - xkPsi
        AE += gam * ( np.dot( z , (psiTmp.T ) ) - AE  )
        r = -1 * np.dot( np.linalg.pinv(AE), b )

        self.z, self.b, self.r, self.AE, self.alpha = z, b, r, AE, alpha

    def Actor(self, xkp1, ukp1, xkp1Psi):
        # ------ Actor -------
        r, D, theta, c, k = self.r, self.D, self.theta, self.c, self.k

        xkp1Psi = xkp1Psi.reshape(-1, 1)
        normR = np.linalg.norm(r)
        tao = ( D / (normR + 0.0) ) if (normR > D) else 1
        beta = 1 if (k <= 1) else ( (c + 0.0 ) / ( k * np.log(k) ) )

        theta = theta - beta * tao * np.dot(r.T, xkp1Psi) * xkp1Psi
        # theta = theta + beta * tao * np.dot(r.T, xkp1Psi) * xkp1Psi
        self.theta = theta

    def _updateWeights(self, xk, uk, gk, xkp1, ukp1, xkPsi, xkp1Psi):
        self.Critic(xk, uk, gk, xkp1, ukp1, xkPsi, xkp1Psi)
        self.Actor(xkp1, ukp1, xkp1Psi)
        self.k += 1


    def _updateWeights_orig(self, xk, uk, gk, xkp1, ukp1, xkPsi, xkp1Psi):
        """It is essentiall the Critic & Actor Step in Actor Critic Algorithm"""
        # ----- Critic --------
        z, b, r, AE, lamb, alpha, k = self.z, self.b, self.r, self.AE, self.lamb, self.alpha, self.k
        # print 'k, ', k
        xkPsi = xkPsi.reshape(-1, 1)
        xkp1Psi = xkp1Psi.reshape(-1, 1)
        z = lamb * z + xkPsi
        gam = 1.0 / (k+1)
        alpha += gam * (gk - alpha)

        b += gam * ( z * ( gk - alpha) - b )
        psiTmp = xkp1Psi - xkPsi
        AE += gam * ( np.dot( z , (psiTmp.T ) ) - AE  )
        r = -1 * np.dot( np.linalg.pinv(AE), b )
        k += 1

        self.z, self.b, self.r, self.AE, self.alpha, self.k = z, b, r, AE, alpha, k

        # ------ Actor -------
        r, D, theta, c = self.r, self.D, self.theta, self.c

        normR = np.linalg.norm(r)
        tao = ( D / (normR + 0.0) ) if (normR > D) else 1
        beta = 1 if (k <= 1) else ( (c + 0.0 ) / ( k * np.log(k) ) )

        theta = theta - beta * tao * np.dot(r.T, xkp1Psi) * xkp1Psi
        # theta = theta + beta * tao * np.dot(r.T, xkp1Psi) * xkp1Psi
        self.theta = theta

        # theta = theta - beta * tao * r

    def learnOnDataSet(self, dataset):
        self.dataset = dataset
        # print 'sequence, ', self.dataset.getLength()
        for n in range(self.dataset.getLength()):
            obs, action, reward = self.dataset.getLinked(n)
            seqidx = ravel(self.dataset['sequence_index'])
            if n == self.dataset.getLength() - 1:
                # last sequence until end of dataset
                loglh = self.loglh['loglh'][seqidx[n], :]
            else:
                loglh = self.loglh['loglh'][seqidx[n]:seqidx[n + 1], :]

            if self.lastobs is not None:
                # print 'lobs, ', self.lastobs, 'laction, ', self.lastaction, 'reward, ', self.lastreward
                # print 'obs, ', obs, 'action, ', action, 'reward, ', reward
                self._updateWeights(self.lastobs, self.lastaction, self.lastreward, obs, action[0], self.lastloglh, loglh)
            self.lastobs = obs
            self.lastaction = action[0]
            self.lastreward = reward
            self.lastloglh = loglh
            # self.learner.newEpisode()



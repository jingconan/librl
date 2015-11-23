#!/usr/bin/env python
# from util import *
import sys
sys.path.append("..")
from librl.util import Expect
from numpy import array, exp, zeros, arange, dot, eye

from pybrain.structure.modules.module import Module
from pybrain.structure.parametercontainer import ParameterContainer
import scipy


class PolicyInterface(object):
    """Interface for policy
    Policy is one type of controller that maps the observation
    to action probability
    PolicyController has two functions:
    1. Generate Actions Based on Observations and Weights
    2. Calculate basis function value. which is nabla_theta psi_theta(x,u)

    """
    def calBasisFuncVal(self, feaList):
        pass

    def calSecondBasisFuncVal(self, feaList):
        pass

class BoltzmanPolicy(Module, ParameterContainer, PolicyInterface):
    """
    for bolzman distribution
            mu(u_i | x) = a_i(theta) / sum(a_i(theta))
            a_i(theta) = F_i(x) exp(
                theta_1 * E{safety( f(x,u_i) )} +
                theta_2 * E{progress( f(x,u_i) )} )
    """

    def __init__(self, feaDim, numActions, T, iniTheta, **args):
        Module.__init__(self, feaDim * numActions, 1, **args)
        ParameterContainer.__init__(self, feaDim)
        self.T = T
        self.PU = None # PU is cached to accelarate the program
        self.g = None
        self.bf = None
        self.theta = iniTheta

        self.numActions = numActions
        self.feaDim = feaDim

        # this two indicators help to make sure the call of first order
        # and second order is synchronized.
        # We make sure firstCallNum- 1 <= secondCallNum < firstCallNum
        self.firstCallNum = 0
        self.secondCallNum = 0

    def get_theta(self): return self._params
    def set_theta(self, val): self._setParameters(val)
    theta = property(fget = get_theta, fset = set_theta)
    params = theta

    def _forwardImplementation(self, inbuf, outbuf):
        """ take observation as input, the output is the action
        """
        n = len(self.theta)
        feature = list(inbuf.reshape(-1, n))
        action_prob = self._getActionProb(feature, self.theta)
        assert self.numActions == len(action_prob), ('wrong number of ',
                                                     'action in policy')
        action = scipy.random.choice(range(self.numActions), action_prob)
        outbuf[0] = array([])

    @staticmethod
    def getActionScore(score, theta, T):
        """Calculate the total score for a control. It is the exp of the
        weighted sum of different features.
        """
        perfer = sum( s * t for s, t in zip(score, theta) )
        return float(exp(perfer/T))

    def _getActionProb(self, feaList, theta):
        """Calculate the Action Probability for each control.
        *feaList* is a list container different feature
        *theta* is the weight for each feature
        """
        scores = scipy.zeros(len(feaList))
        for i, feature in enumerate(feaList):
            scores[i] = self.getActionScore(feature, theta, self.T)

        return scores / scipy.sum(scores)

    def getActionValues(self, obs):
        """extract features from observation and call _getActionProb"""
        return array(self._getActionProb(self.obs2fea(obs), self.theta))

    def obs2fea(self, obs):
        """observation to feature list"""
        n = len(self.theta)
        return list(obs.reshape(-1, n))
    def fea2obs(self, fea):
        """feature list to observation"""
        return fea.reshape(-1)

    def calBasisFuncVal(self, feaList):
        """for an observation, calculate value of basis function
        for all possible actions
            feaList is a list of tuple. each tuple represent the value of feature
            take the robot motion control as an example, a possible value may be:
                [
                ( safety_1 , progress_1),
                ( safety_2 , progress_2),
                ( safety_3 , progress_3),
                ( safety_4 , progress_4),
                ]
                1, 2, 3, 4 coressponds to each action ['E', 'N', 'W', 'S']
            ]

            Basis Function Value: is the first order derivative of the log of the policy.
        """
        self.firstCallNum += 1
        feaMat = scipy.array(feaList)
        action_prob = self._getActionProb(feaList, self.theta)
        self.g = scipy.dot(action_prob, feaMat)
        self.bf = feaMat - self.g
        return self.bf

    def calSecondBasisFuncVal(self, feaList):
        """ calculate \nab^2 log(\mu)
        Please see https://goo.gl/PRnu58 for mathematical deduction.
        """
        feaMat = scipy.array(feaList)
        action_prob = self._getActionProb(feaList, self.theta)
        mat1 = scipy.dot(feaMat.T * action_prob, feaMat)
        tmp = scipy.dot(feaMat.T, action_prob)
        log_likelihood_hessian = -1 * mat1 + scipy.outer(tmp, tmp.T)
        return log_likelihood_hessian
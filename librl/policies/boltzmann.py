import scipy
from scipy import array, exp, zeros, arange, dot, eye

from pybrain.structure.modules.module import Module
from pybrain.structure.parametercontainer import ParameterContainer
from pybrain.utilities import abstractMethod

class PolicyInterface(object):
    """Interface for policy
    Policy is one type of controller that maps the observation
    to action probability
    PolicyController has two functions:
    1. Generate Actions Based on Observations and Weights
    2. Calculate basis function value. which is nabla_theta psi_theta(x,u)

    """
    def calBasisFuncVal(self, feaList):
        abstractMethod()

    def calSecondBasisFuncVal(self, feaList):
        abstractMethod()

class BoltzmanPolicy(Module, ParameterContainer, PolicyInterface):
    """
    for bolzman distribution
            mu(u_i | x) = a_i(theta) / sum(a_i(theta))
            a_i(theta) = F_i(x) exp(
                theta_1 * E{safety( f(x,u_i) )} +
                theta_2 * E{progress( f(x,u_i) )} )
    """

    def __init__(self, actionnum, T, theta, **args):
        self.feadim = len(theta)
        Module.__init__(self, self.feadim * actionnum, 1, **args)
        ParameterContainer.__init__(self, self.feadim)
        self.T = T
        self.g = None
        self.bf = None

        # feadimx1 vector.
        self.theta = theta
        self.actionnum = actionnum

    def get_theta(self): return self._params
    def set_theta(self, val): self._setParameters(val)
    theta = property(fget = get_theta, fset = set_theta)
    params = theta

    def _forwardImplementation(self, inbuf, outbuf):
        """ take observation as input, the output is the action
        """
        action_prob = self._getActionProb(self.obs2fea(inbuf), self.theta)
        assert self.actionnum == len(action_prob), ('wrong number of ',
                                                     'action in policy')
        action = scipy.random.choice(range(self.actionnum), p=action_prob)
        outbuf[0] = action

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
        return obs.reshape(self.actionnum, self.feadim)

    def fea2obs(self, fea):
        """feature list to observation"""
        obs = fea.reshape(-1)
        assert len(obs) == self.actionnum * self.feadim, 'invalid feature!'
        return obs

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


class PolicyFeatureModule(Module):
    """Module to calculate features for state-action value function approximiation.
    Input: a vector whose the last element is the action, and the rest elements are
           observation.
    Output: a vector of features which is used for approximate state-action
    value function.
    """
    def __init__(self, policy, name=None):
        self.policy = policy
        self.feadim = policy.feadim
        self.actionnum = policy.actionnum
        self.featuredescriptor = self.getFeatureDescritor()

        outdim = 0
        for feature in self.featuredescriptor:
            outdim += feature['dimension']

        super(PolicyFeatureModule, self).__init__(
            indim = self.feadim * self.actionnum + 1,
            outdim = outdim,
            name = name
        )

    def getFeatureDescritor(self):
        def _firstOrderFeature(policy, feature, action):
            return self.getFeatureSlice(policy.calBasisFuncVal(feature).reshape(-1),
                                        action)

        def _secondOrderFeature(policy, feature, action):
            return policy.calSecondBasisFuncVal(feature).reshape(-1)

        return [
            {
                'dimension': self.feadim,
                'constructor': _firstOrderFeature,
            },
            {
                'dimension': self.feadim * self.feadim,
                'constructor': _secondOrderFeature,
            },
        ]

    def getFeatureSlice(self, feature, action):
        featureslice = self.policy.obs2fea(feature[0:(self.feadim *
                                                      self.actionnum)])
        return featureslice[action, :]

    def _forwardImplementation(self, inbuf, outbuf):
        fea = self.policy.obs2fea(inbuf[:-1])
        action = inbuf[-1]
        offset = 0
        for desc in self.featuredescriptor:
            newoffset = offset + desc['dimension']
            outbuf[offset:newoffset] = desc['constructor'](self.policy,
                                                           fea, action)
            offset = newoffset

    def get_theta(self): return self.policy.theta.reshape(-1)
    def set_theta(self, val): self.policy._setParameters(val.reshape(-1))
    theta = property(fget = get_theta, fset = set_theta)


class PolicyValueFeatureModule(PolicyFeatureModule):
    """Module to generate feature for approximating state value function

    The first part of the output is the _firstOrderFeature as in
    PolicyFeatureModule. The second part is value function is appended.
    """

    def getFeatureDescritor(self):
        self.statefeadim = int(self.feadim / self.actionnum)

        def _firstOrderFeature(policy, feature, action):
            return self.getFeatureSlice(policy.calBasisFuncVal(feature).reshape(-1),
                                        action)

        def _stateFeature(policy, feature, action):
            return feature[0][:self.statefeadim]

        return [
            {
                'dimension': self.feadim,
                'constructor': _firstOrderFeature,
            },
            {
                'dimension': self.statefeadim,
                'constructor': _stateFeature,
            },
        ]

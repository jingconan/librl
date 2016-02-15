import copy
import scipy
from scipy import array, exp, zeros, arange, dot, eye

from librl.util import encodeTriuAs1DArray,decode1DArrayAsSymMat
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

        self.cachedActionProb = None

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
        self.cachedActionProb = action_prob
        self.g = scipy.dot(action_prob, feaMat)
        self.bf = feaMat - self.g
        return self.bf

    def calSecondBasisFuncVal(self, feaList):
        """ calculate \nab^2 log(\mu)
        Please see https://goo.gl/PRnu58 for mathematical deduction.
        """
        feaMat = scipy.array(feaList)
        # We assume second order basis calculation followe first order basis
        # calculation, so we just use the cachedActionProb.
        if self.cachedActionProb is not None:
            action_prob = self.cachedActionProb
            self.cachedActionProb = None
        else:
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

        self.feadesc, outdim = self._transformFeatureDescriptor(
            self.getFeatureDescriptor())

        super(PolicyFeatureModule, self).__init__(
            indim = self.feadim * self.actionnum + 1,
            outdim = outdim,
            name = name
        )

    def _transformFeatureDescriptor(self, feadesc):
        # parse feature descriptor.
        bd = [0] # boundary
        feanames = []
        outdim = 0
        for feature in feadesc:
            outdim += feature['dimension']
            feanames.append(feature['name'])
            bd.append(outdim)

        newfeadesc = dict()
        for i, name in enumerate(feanames):
            desc = copy.deepcopy(feadesc[i])
            desc['fea_range'] = (bd[i], bd[i+1])
            desc['fea_index'] = i
            newfeadesc[name] = desc
        return newfeadesc, outdim

    def decodeFeature(self, feature, name):
        decoder = self.feadesc[name].get('decoder', self._identityDecoder)
        r = self.feadesc[name]['fea_range']
        feature = feature[r[0]:r[1]]
        return decoder(feature)

    # the default decoder.
    @staticmethod
    def _identityDecoder(feature):
        return feature

    def getFeatureDescriptor(self):
        # first order feature
        def _firstOrderFeature(policy, feature, action):
            return self.getFeatureSlice(policy.calBasisFuncVal(feature).reshape(-1),
                                        action)

        # second order feature
        n = self.feadim
        sofl = (n * n - n) / 2 + n # second order feature length

        def _secondOrderFeature(policy, feature, action):
            hessian = policy.calSecondBasisFuncVal(feature)
            return encodeTriuAs1DArray(hessian)

        def _secondOrderFeatureDecoder(feature):
            assert sofl == len(feature), ('invalid feature'
                                          'to decode')
            return decode1DArrayAsSymMat(feature, self.feadim)

        return [
            {
                'name': 'first_order',
                'dimension': self.feadim,
                'constructor': _firstOrderFeature,
                'decoder': self._identityDecoder,
            },
            {
                'name': 'second_order',
                'dimension': sofl,
                'constructor': _secondOrderFeature,
                'decoder': _secondOrderFeatureDecoder,
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
        for name, desc in self.feadesc.iteritems():
            fearange = desc ['fea_range']
            outbuf[fearange[0]:fearange[1]] = desc['constructor'](self.policy,
                                                           fea, action)

    def get_theta(self): return self.policy.theta.reshape(-1)
    def set_theta(self, val): self.policy._setParameters(val.reshape(-1))
    theta = property(fget = get_theta, fset = set_theta)


class PolicyValueFeatureModule(PolicyFeatureModule):
    """Module to generate feature for approximating state value function

    The first part of the output is the _firstOrderFeature as in
    PolicyFeatureModule. The second part is value function is appended.
    """

    def getFeatureDescriptor(self):
        self.statefeadim = int(self.feadim / self.actionnum)

        def _firstOrderFeature(policy, feature, action):
            return self.getFeatureSlice(policy.calBasisFuncVal(feature).reshape(-1),
                                        action)

        def _stateFeature(policy, feature, action):
            return feature[0][:self.statefeadim]

        return [
            {
                'name': 'first_order',
                'dimension': self.feadim,
                'constructor': _firstOrderFeature,
            },
            {
                'name': 'state_feature',
                'dimension': self.statefeadim,
                'constructor': _stateFeature,
            },
        ]

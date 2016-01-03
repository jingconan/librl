"""This files contains the method described in paper
    Natural Actor-Critic Algorithms. S. Bhatnagar, RS. Sutton,
    M. Ghavamzadeh, and M Lee. https://webdocs.cs.ualberta.ca/~sutton/papers/BSGL-TR.pdf
"""
from __future__ import print_function, division, absolute_import
import scipy
from scipy.linalg import norm
from .actorcritic import ActorCriticLearner

class BSGLRegularGradientActorCriticLearner(ActorCriticLearner):
    """Regular gradient"""

    def __init__(self,
                 cssinitial, cssdecay, # css means critic step size
                 assinitial, assdecay, # ass means actor steps size
                 rdecay, # reward decay weight
                 parambound, # bound for the parameters
                 maxcriticnorm, # maximum critic norm
                 *args, **kwargs):
        super(BSGLRegularGradientActorCriticLearner, self).__init__(*args,
                                                                    **kwargs)
        self.cssinitial = cssinitial
        self.cssdecay = cssdecay
        self.assinitial = assinitial
        self.assdecay = assdecay
        self.rdecay = rdecay
        self.maxcriticnorm = maxcriticnorm
        if parambound is None:
            self.parambound = None
        else:
            self.parambound = scipy.array(parambound)


    def reset(self):
        """reset all parameters"""
        self.resetStepSize()
        # state feature dimension
        self.sfdim = self.module.feadim * self.module.actionnum
        self.r = scipy.zeros((self.sfdim,))
        self.alpha = 0
        self.lastobs = None

    def newEpisode(self):
        """new Episode only restart the counter,
        not the parameter that has been estimated"""
        self.k = 0
        self.lastobs = None

    def critic(self, lastreward, lastfeature, reward, feature):
        # Get state features
        slastfeature = lastfeature[-self.sfdim:]
        sfeature = feature[-self.sfdim:]

        # Estimate of avg reward.
        # reward learning rate = reward decay factor x critic step size.
        rweight = self.rdecay * self.gamma()
        self.alpha = (1 - rweight) * self.alpha + rweight * reward

        # Update critic parameter
        self.d = reward - self.alpha + scipy.inner(self.r, sfeature - slastfeature)
        self.r += self.gamma() * self.d * sfeature

        normr = norm(self.r)
        if normr > self.maxcriticnorm:
            self.r = self.r / normr * self.maxcriticnorm

    # critic step size
    def gamma(self):
        return self.cssinitial * self.cssdecay / (self.cssdecay + self.k **
                                                  (2.0 / 3))

    # actor step size
    def beta(self):
        return self.assinitial * self.assdecay / (self.assdecay + self.k)

    def ensureBound(self, v):
        if self.parambound is None:
            return v
        return scipy.clip(v, self.parambound[:, 0], self.parambound[:, 1])

    def actor(self, obs, action, feature):
        update = self.beta() * self.d * feature[:self.paramdim]
        self.module.theta = self.ensureBound(self.module.theta + update)

class BSGLFisherInfoActorCriticLearner(BSGLRegularGradientActorCriticLearner):
    INITIAL_IFIM = 1.5

    def reset(self):
        """reset all parameters"""
        super(BSGLFisherInfoActorCriticLearner, self).reset()
        # inverse fisher information matrix.
        self.ifim = self.INITIAL_IFIM * scipy.eye(self.paramdim)

    def critic(self, lastreward, lastfeature, reward, feature):
        super(BSGLFisherInfoActorCriticLearner, self).critic(lastreward,
                                                             lastfeature,
                                                             reward, feature)
        # Here we use Sherman-Morrison matrix inversion lemma.
        # A 0.001 scaling factor is used for numerical stability.
        css = 0.001 * self.gamma()
        psi = feature[:self.paramdim]
        tmp = scipy.inner(self.ifim, psi)
        update = scipy.outer(tmp, tmp) / (1 - css + css * scipy.inner(psi, tmp))
        self.ifim = (1.0 / (1 - css)) * (self.ifim - css * update)

    def actor(self, obs, action, feature):
        update = self.beta() * self.d * scipy.inner(self.ifim, feature[:self.paramdim])
        self.module.theta = self.ensureBound(self.module.theta + update)

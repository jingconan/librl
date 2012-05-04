from LSTDACLearner import *
import copy



class SecondBasisDataSet(DataSet):
    def __init__(self, dim):
        DataSet.__init__(self)
        self.addField('SB', dim)
        self.linkFields(['SB'])
        self.index = 0

class HessianACLearner(LSTDACLearner):
    """
    ############################### Description Of Parameter ################################
    # psi is a N by 1 vector representing the gradient of policy mu
    # varsigma is a N by N matrix representing the hessian matrix of policy mu
    # r is the projection of Q on the space spanned by {psi}, r is N by 1, where
    #       n is dimension of theta
    # S is the projection of Q on the spance spanned by varsigma, N by N. the mth
    #       column Sm is the projection of Q on the mth row of varsigma
    # TE is the projection of gradient Q on the space of {psi}, N by N. Gradient Q
    #       is a n by 1 vector. the mth colum si the projection of mth component of Q on {psi}
    ############################ END Description Of Parameter ################################
    # Global Variance HessianAC used
    # n: dimension of theta
    # gridSize:
    # uSize: number of all possible controls
    """
    def __init__(self,  actiondim, iniTheta, **argv):
        print 'Hessian'
        LSTDACLearner.__init__(self, actiondim, iniTheta, **argv)
        n = len(iniTheta)
        gridSize = argv['gridSize']
        uSize = argv['uSize']
        self.hessianThetaTh  = argv['hessianThetaTh']
        self.TE = np.zeros([n, n])
        # self.varsigma = np.zeros( [gridSize[0], gridSize[1], uSize, n, n] )
        self.y = np.zeros([n, n]) # Counterparts of z
        self.a = np.zeros([n, n]) # counterparts of b
        self.S = np.zeros([n, n]) # counterparts of r, coef for Q on varsigma
        self.EE = np.zeros([n, n, n]) # counterparts of AE.
        self.h = np.zeros([gridSize[0], gridSize[1], uSize, n])
        self.v = np.zeros([n, n]) # counterparts of b
        self.t = np.zeros([n, n]) # counterparts of r, coef for grad Q on psi

        self.gradient = np.zeros([n, 1])
        self.hessian = np.zeros([n, n])

        self.hessianRec = []

        self.SB = SecondBasisDataSet(self.feadim ** 2)

    def Critic(self, xk, uk, gk, xkp1, ukp1, xkPsi, xkp1Psi, xkVarsigma, xkp1Varsigma):
        LSTDACLearner.Critic(self, xk, uk, gk, xkp1, ukp1, xkPsi, xkp1Psi)
        # print 'finish old critic'
        k = self.k
        n = self.feadim
        xkPsi = xkPsi.reshape(-1, 1)
        xkp1Psi = xkp1Psi.reshape(-1, 1)
        xkVarsigma.shape = (n, n)
        xkp1Varsigma.shape = (n, n)

        gam = 1.0 / (k+1)
        # Estimate h(xk, uk)
        # print 'xkPsi, ', xkPsi
        # import pdb;pdb.set_trace()
        self.h[xk[0], xk[1], uk, :] = np.dot( self.r.T, xkPsi ) * xkPsi.reshape(-1)
        # Estimate Projection of Q on varsigma
        for m in xrange(n): # For each Component of varsigma
            # self.y[:, m] = self.lamb * self.y[:, m]  + self.varsigma[xk[0], xk[1], uk, :, m]
            # import pdb;pdb.set_trace()
            self.y[:, m] = self.lamb * self.y[:, m]  +xkVarsigma[:, m]
            self.a[:, m] = self.a[:, m] + gam * ( gk * self.y[:, m] - self.a[:, m] )
            tmp = xkp1Varsigma[ :, m] - xkVarsigma[:, m]
            self.EE[m, :, :] = self.EE[m, :, :] + gam * ( np.dot( self.y[:, m].reshape(-1, 1), tmp.reshape(1, -1)) - self.EE[m, :, :] )

            self.S[:, m] = -1 * np.dot( np.linalg.pinv(self.EE[m, :, :]), self.a[:, m])

        # Estimate Projection of grad Q on psi
        for m in xrange(n):
            self.v[:, m] += gam * (self.h[xk[0], xk[1], uk, m] * self.z.reshape(-1,) - self.v[:, m]) #FIXME, the shape of z is not elegent
            self.t[:, m] = -1 * np.dot( np.linalg.pinv(self.AE), self.v[:, m])

    def Actor(self, xkp1, ukp1, xkp1Psi, xkp1Varsigma):
        r, D, theta, c, k = self.r, self.D, self.theta, self.c, self.k #FIXME Dirty Code
        n = self.feadim
        xkp1Psi = xkp1Psi.reshape(-1, 1)
        xkp1Varsigma = xkp1Varsigma.reshape(n, n)

        normR = np.linalg.norm(r)
        tao = ( (c + 0.0 ) / ( (k+1) * np.log(k+1) ) ) if (normR > D) else 1
        beta = 0 if (k == 0) else (c + 0.0 ) / ( (k+1) * np.log(k+1) )

        thetaBefore = copy.deepcopy(theta)
        gradLambda = np.dot(r.T, xkp1Psi) * xkp1Psi

        iota = 0.7
        gradientBefore = copy.deepcopy( self.gradient )
        self.gradient = iota * self.gradient + gradLambda
        # Get Hessian For Lambda
        hessian = np.zeros([n, n])
        for i in xrange(n):
            for j in xrange(n):
                QPrjCoef = np.dot( self.S[: ,i], xkp1Varsigma[:, i] )
                gradQPrjCoef = np.dot(self.t[:, j], xkp1Psi)
                hessian[i, j] =   QPrjCoef * xkp1Varsigma[i, j] + gradQPrjCoef * xkp1Psi[i]

        hessianBefore = copy.deepcopy( self.hessian )
        self.hessian = iota * self.hessian + hessian
        # import pdb;pdb.set_trace()
        theta = theta - beta * tao * np.dot( np.linalg.pinv(self.hessian),  self.gradient )
        # theta = theta - beta * tao * np.dot( np.linalg.pinv(hessian),  gradLambda )

        thetaDiff = sum(abs(theta - thetaBefore ))
        # returnFlag = True if ( thetaDiff < th) else False
        if thetaDiff > self.hessianThetaTh:
            print 'wierd'
            theta = thetaBefore
            self.gradient = gradientBefore
            self.hessian = hessianBefore
            thetaDiff = 0

        self.theta = theta
        # return returnFlag, thetaDiff

    def _updateWeights(self, xk, uk, gk, xkp1, ukp1, xkPsi, xkp1Psi, xkVarsigma, xkp1Varsigma):
        self.Critic(xk, uk, gk, xkp1, ukp1, xkPsi, xkp1Psi, xkVarsigma, xkp1Varsigma)
        self.Actor(xkp1, ukp1, xkp1Psi, xkp1Varsigma)
        self.k += 1


    def learnOnDataSet(self, dataset):
        self.dataset = dataset
        for n in range(self.dataset.getLength()):
            obs, action, reward = self.dataset.getLinked(n)
            seqidx = ravel(self.dataset['sequence_index'])
            if n == self.dataset.getLength() - 1:
                # last sequence until end of dataset
                loglh = self.loglh['loglh'][seqidx[n], :]
                secondBasis = self.SB['SB'][seqidx[n], :]
            else:
                loglh = self.loglh['loglh'][seqidx[n]:seqidx[n + 1], :]
                secondBasis = self.SB['SB'][seqidx[n]:seqidx[n + 1], :]

            if self.lastobs is not None:
                self._updateWeights(self.lastobs, self.lastaction, self.lastreward, obs, action[0], self.lastloglh, loglh, self.lastSecondBasis, secondBasis)
            self.lastobs = obs
            self.lastaction = action[0]
            self.lastreward = reward
            self.lastloglh = loglh
            self.lastSecondBasis = secondBasis

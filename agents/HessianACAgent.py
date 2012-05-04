from LSTDACAgent import *

from numpy import ravel
class HessianACAgent(LSTDACAgent):
    def getAction(self):
        """This is basically the Actor part"""
        LoggingAgent.getAction(self)
        self.lastaction = self.policy.activate(self.feaList, self.learner.theta)
        self.lastBasisFuncValue = self.policy.calBasisFuncVal(self.feaList)
        self.lastSecondBasisFuncValue = self.policy.calSecondBasisFuncVal(self.feaList)
        return self.lastaction

    def giveReward(self, r):
        LSTDACAgent.giveReward(self, r)
        if self.logging:
            self.learner.loglh.appendLinked(self.lastBasisFuncValue[self.lastaction[0]])
            self.learner.SB.appendLinked(ravel(self.lastSecondBasisFuncValue[self.lastaction, :, :]))


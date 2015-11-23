
class Process(object):
    def _environment(self):
        envMatrix = zeros(gridSize)
        envMatrix[zip(*unsafeStates)] = -1
        return TrapMaze(envMatrix, iniState, goalStates, TP, DF)

    def _policy(self):
        from policy import BoltzmanPolicy
        return BoltzmanPolicy(feaDim = 2, numActions = 4, T = T, iniTheta=iniTheta)

    def _task(self):
        from task import RobotMotionTask
        self.env = self._environment
        return RobotMotionTask(selef.env, senRange=senRange)

    def _leaner(self):
        pass

    def _init_trace(self):
        pass

    def export_trace(self):
        pass

    def check(self):
        """user defined check process"""
        pass

    def _agent(self):
        policy = self._policy()
        learner = self._learner()
        return ACAgent(policy, learner, sdim=8, adim=1)

    def _experiment(self):
        self.task = self._task()
        self.agent = self._agent()
        return Experiment(self.task, self.agent)

    def loop(self):
        experiment = self._experiment()
        self._init_trace()
        r = 0
        j = -1
        while True:
            reward = experiment._oneInteraction()
            r += reward
            self.agent.learn()
            if self.check(**locals()):
                break
        self.export_trace()

    def loop_exception_handle(self):
        try:
            self.loop()
        except KeyboardInterrupt:
            pass

from ReachProbCalculator import ReachProbCalculator
class HACProcess(Process):
    def _leaner(self):
        from learners import HACLearner
        return HACLearner(lamb = 0.9, c = 10, D=20)

    def _init_trace(self):
        self.trace = dict(
                    ep=[], # episode number
                    reward=[], # reward of this epsisode
                    it=[], # total iteration number
                    theta0=[], # first value of theta
                    theta1=[], # second value of theta
                    )
        # more detailed record for theta value
        self.th_trace = dict(
                theta0=[],
                theta1=[],
                it=[],
                )
        self.reachProb = ReachProbCalculator(self.env, self.task, self.agent)

    def export_trace(self):
        from util import WriteTrace
        WriteTrace(trace, 'hac.tr')
        WriteTrace(th_trace, 'hac_theta.tr')

    def check(self, j, **kwargs):
        if j == 1e6:
            return True

        if j % 100 == 0:
            print 'iter: [', j, '] theta value: [%f, %f]'%tuple(policy.theta)
            self.th_trace['theta0'].append(policy.theta[0])
            self.th_trace['theta1'].append(policy.theta[1])
            self.th_trace['it'].append(j)

        if j % 3e3 == 0:
            rp, time = self.reachProb.GetReachProb(policy.theta)
            print 'rp, ', rp, 'time, ', time
            if rp > 0.5:
                return True

        if task.reachGoalFlag:
            i += 1
            # if i == 5e3: break
            trace['theta0'].append(policy.theta[0])
            trace['theta1'].append(policy.theta[1])
            trace['ep'].append(i)
            trace['reward'].append(r)
            trace['it'].append(j)
            print '[%i]reach goal, reward'%(i), r
            r = 0
            task.reset()
            learner.resetStepSize()

        return False

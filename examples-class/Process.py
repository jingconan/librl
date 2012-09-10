import sys
sys.path.insert(0, "..")
###############################
##       Parameters          ##
###############################
# import global parameters
from problem_settings import gridSize, unsafeStates, iniState, goalStates, TP, DF, senRange
from problem_settings import T, iniTheta

##############################################
##    Import Usefule Modules and Functins   ##
##############################################
from scipy import zeros

class Process(object):
    def _environment(self):
        from environments import TrapMaze
        envMatrix = zeros(gridSize)
        envMatrix[zip(*unsafeStates)] = -1
        return TrapMaze(envMatrix, iniState, goalStates, TP, DF)

    def _policy(self):
        from policy import BoltzmanPolicy
        return BoltzmanPolicy(feaDim = 2, numActions = 4, T = T, iniTheta=iniTheta)

    def _task(self):
        from task import RobotMotionTask
        self.env = self._environment()
        return RobotMotionTask(self.env, senRange=senRange)

    def _learner(self):
        pass

    def _init_trace(self):
        pass

    def export_trace(self):
        pass

    def check(self):
        """user defined check process"""
        pass

    def _agent(self):
        from agents import ACAgent
        self.policy = self._policy()
        self.learner = self._learner()
        return ACAgent(self.policy, self.learner, sdim=8, adim=1)

    def _experiment(self):
        from pybrain.rl.experiments import Experiment
        self.task = self._task()
        self.agent = self._agent()
        return Experiment(self.task, self.agent)

    def loop(self):
        experiment = self._experiment()
        self._init_trace()
        self.r = 0
        self.j = -1
        while True:
            self.j += 1
            reward = experiment._oneInteraction()
            self.r += reward
            self.agent.learn()
            if self.check():
                break
        self.export_trace()

    def loop_exception_handle(self):
        try:
            self.loop()
        except KeyboardInterrupt:
            self.export_trace()

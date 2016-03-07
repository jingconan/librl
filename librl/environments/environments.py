import scipy
from pybrain.rl.environments.task import Task

class StateObsWrapperTask(Task):
    """A wrapper task that adds state observations to tasks that only output
       state-action observations"""
    def __init__(self, task):
        self.task = task
        self.numActions = self.task.env.numActions

    @property
    def outdim(self):
        edim = self.task.env.outdim
        return self.task.outdim + edim

    def performAction(self, action):
        return self.task.performAction(action)

    def getReward(self):
        return self.task.getReward()

    def getObservation(self):
        edim = self.task.env.outdim
        feature = scipy.zeros((self.numActions + 1, edim))
        obs = self.task.getObservation()
        feature[:self.numActions, :] = obs.reshape((self.numActions, edim))

        sensors = self.task.env.getSensors()
        feature[self.numActions, :] = scipy.array(sensors)

        return feature.reshape(-1)



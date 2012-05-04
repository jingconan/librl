'''
The controller in PyBrain is a module that takes states as
inputs and transforms them into actions.The agent consists of a controller,
a learner and an explorer.
The learner updates the controller parameters according to the
interaction it had with the world. The explorer add some exploration
to the action get by controller.
'''


class PGACController(object):
    '''PGACController is the controller for Policy Gradient
    Actor-Critic Method.

    '''

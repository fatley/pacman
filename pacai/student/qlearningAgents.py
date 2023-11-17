from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection
from pacai.util.probability import flipCoin
import random

class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:
    Get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

        # You can initialize Q-values here.
        self.qValues = {}

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """

        # getQValue just returns the qVal of specific state action pair from dict
        if (state, action) in self.qValues:
            # if in dict, return qval
            return self.qValues[(state, action)]
        else:
            # never seen, return 0.0
            return 0.0

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """

        # getValue does max(Q(s', a')
        # if no legal actions (tstate), return 0.0
        # if there are legal actions in state, return max qval
        if len(self.getLegalActions(state)) != 0:
            return max([self.getQValue(state, action) for action in self.getLegalActions(state)])
        else:
            return 0.0

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """

        # getPolicy returns the action with the max qval
        # instead of returning value, return action
        # if no legal actions (tstate), return None
        # if there are legal actions in state, return action with max qval
        if len(self.getLegalActions(state)) != 0:
            return max([(self.getQValue(state, action), action)
                        for action in self.getLegalActions(state)])[1]
        else:
            return None

    def update(self, state, action, nextState, reward):
        """"
        The parent class calls this to observe a state transition and reward.
        You should do your Q-Value update here.
        Note that you should never call this function, it will be called on your behalf.
        """
        # newQ(s, a) = Q(s,a) + alpha[R(s,a) + gamma * maxQ'(s',a') - Q(s,a)]
        # alpha = learning rate
        # gamma = discount rate
        # maxQ'(s',a') = max qval of next state
        # Q(s,a) = qval of current state
        # R(s,a) = reward of current state

        # sample = R(s,a) + gamma * maxQ'(s',a')
        
        sample = reward + self.discountRate * self.getValue(nextState)
        newQVal = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample
        self.qValues[(state, action)] = newQVal
        
    def getAction(self, state):
        """
        Compute the action to take in the current state.
        With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
        we should take a random action and take the best policy action otherwise.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should choose None as the action.
        """

        # calculate action with probability epsilon
        # take a random action and take best policy action otherwise
        # if no legal actions, return None
        # epilson - exploration rate
        # can choose an element from a list uniformly at random by calling the random.choice
        # can simulate a binary variable with probability p of success
        # by using pacai.util.probability.flipCoin,
        # which returns True with probability p and False with probability 1 - p.

        if len(self.getLegalActions(state)) == 0:
            return None
        
        # take a random action with probability epsilon
        # with probability r, act randomly
        # with probability 1-r, act according to current policy
        # flip coin = prob: true - p, false- 1-p
        if flipCoin(self.epsilon):
            return random.choice(self.getLegalActions(state))
        else:
            return self.getPolicy(state)
        
class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action

class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index,
            extractor = 'pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)

        # You might want to initialize weights here.

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            # You might want to print your weights here for debugging.
            # *** Your Code Here ***
            raise NotImplementedError()

from pacai.agents.learning.value import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    # value of a state is the max expected reward over all possible actions
    # where the expected reward for an action is the sum of the immediate reward
    # and the discounted future reward
    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index, **kwargs)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = {}  # A dictionary which holds the q-values for each state.

        # iterations 
        for i in range(iters):
            # store updated values for each state in current iter
            new_values = {}
            # iterating through all states to find the max qvalue
            for state in self.mdp.getStates():
                # if it is a term state, set value to 0
                if self.mdp.isTerminal(state):
                    new_values[state] = 0
                else:
                    # if not term state, get all possible action and calculate qvalue for actions - then grab max qval
                    new_values[state] = max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)])
            # updating values for each state
            self.values = new_values
                    
    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """

        return self.values.get(state, 0.0)
    
    # P(s'|s,a) * [R(s,a,s') + γ * V(s')]
    def getQValue(self, state, action):
        qValue = 0.0
        # getting probability and transition state of the next possible state 
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            # qvalue is sum of probability *  (immediate reward + discount factor * value of next state)
            qValue += prob * (self.mdp.getReward(state, action, nextState) + self.discountRate * self.getValue(nextState))
        return qValue
    
    # best action in a given state, max qvalue
    def getPolicy(self, state):
        possibleActions = self.mdp.getPossibleActions(state)
        # if no possible actions, return None
        if len(possibleActions) == 0:
            return None
        # else return action with max qvalue
        bestAction = max(possibleActions, key=lambda action: self.getQValue(state, action)) 
        return bestAction
        
    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        return self.getPolicy(state)

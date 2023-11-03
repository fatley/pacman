import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core.distance import manhattan
class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # *** Your Code Here ***
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]
        
        # # checking for each ghost
        # # grabbing curr position of pacman & ghost
        # oldPosition = currentGameState.getPacmanPosition()
        # oldGhostStates = currentGameState.getGhostStates()
        # # for each ghost, check if the new position is closer than the old position
        # for oldGhostState, newGhostState in zip(oldGhostStates, newGhostStates):
        #     oldGhostPosition = oldGhostState.getPosition()
        #     newGhostPosition = newGhostState.getPosition()
            
        #     # if the new position is closer than the old position, then return -inf
        #     oldDistance = manhattan(oldPosition, oldGhostPosition)
        #     newDistance = manhattan(newPosition, newGhostPosition)
            
        #     if newDistance < 2 and newDistance < oldDistance:
        #         return float("-inf")
        # # capsules
        # capsules = currentGameState.getCapsules()
        # if capsules:
        #     closestCapsule = min(capsules, key=lambda capsule: manhattan(newPosition, capsule))
        #     for ghostState in newGhostStates:
        #         if manhattan(ghostState.getPosition(), closestCapsule) < 2:
        #             foodList = oldFood.asList()
        #             if foodList:
        #                 closestFood = min([manhattan(newPosition, food) for food in foodList])
        #                 if closestFood == 0:
        #                     return successorGameState.getScore() + 1
        #                 else:
        #                     return successorGameState.getScore() + 1 / closestFood
        
        # # if the ghost is scared, eat everythign in sight while the ghost is scared
        # if newScaredTimes[0] > 0:
        #     foodList = oldFood.asList()
        #     if foodList:
        #         closestFood = min([manhattan(newPosition, food) for food in foodList])
        #         if closestFood == 0:
        #             return successorGameState.getScore() + 1
        #         else:
        #             return successorGameState.getScore() + 1 / closestFood
        #     else:
        #         return successorGameState.getScore() + 1000
        
        # # if there are no capsules, carry on with normal routine
        # # checking for food
        # foodList = oldFood.asList()
        # if foodList:
        #     closestFood = min([manhattan(newPosition, food) for food in foodList])
        #     if closestFood == 0:
        #         return successorGameState.getScore() + 1
        #     else:
        #         return successorGameState.getScore() + 1 / closestFood
    
        # # checking for ghost
        # for ghostState in newGhostStates:
        #     if ghostState.getPosition() < 2:
        #         return float("-inf")
            
        # # if there are no ghost, eat all teh food around
        # if not any(ghostState.getPosition() < 3 for ghostState in newGhostStates):
        #     return successorGameState.getScore() + 1000
        
        # return successorGameState.getScore()
    
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPosition = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # walls
        walls = currentGameState.getWalls()
        x, y = newPosition
        if sum([walls[x+dx][y+dy] for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]) >= 3:
            return float("-inf")

        # Calculate the distance to the closest ghost
        ghostDistances = [manhattan(newPosition, ghostState.getPosition()) for ghostState in newGhostStates]
        closestGhostDistance = min(ghostDistances) if ghostDistances else float("inf")

        # Calculate the distance to the closest food
        foodDistances = [manhattan(newPosition, food) for food in oldFood.asList()]
        closestFoodDistance = min(foodDistances) if foodDistances else float("inf")

        # Calculate the score
        score = successorGameState.getScore()

        # If there's a ghost too close, return a very low score
        if closestGhostDistance < 2:
            return float("-inf")

        # If there's a scared ghost, prioritize eating food
        if newScaredTimes[0] > 0:
            return score + 1 / closestFoodDistance if closestFoodDistance != 0 else score + 1

        # If there are no ghosts too close, prioritize eating food
        if not any(manhattan(newPosition, ghostState.getPosition()) < 3 for ghostState in newGhostStates):
            return score + 1 / closestFoodDistance if closestFoodDistance != 0 else score + 1

        return score   
    
    
class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        
    # get action func
    # returns minimax action from current gameState using getTreeDepth and getEvaluationFunction 
    # getTreeDepth - returns the depth of the search tree
    # getEvaluationFunction - returns the evaluation function for the search problem
    # minimax - returns the minimax value of a state   
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        
        depth = self.getTreeDepth()
        return self.minimax(gameState, depth, 0)[1]
        
    # minimax func
    # returns the minimax value of a state
    def minimax(self, state, depth, agentIndex):
        # if state is terminal, return the state's utility
        if state.isWin() or state.isLose() or depth == 0:
            # return state.evaluationFunction(state), None
            return self.getEvaluationFunction()(state), None
        
        # if agent is pacman, return max value
        if agentIndex == 0:
            return self.maxValue(state, depth, agentIndex)
        
        # if agent is ghost, return min value
        else:
            return self.minValue(state, depth, agentIndex)
        
    # max value func
    # returns the max value of a state
    def maxValue(self, state, depth, agentIndex):
        # if state is terminal, return the state's utility
        if state.isWin() or state.isLose():
            return state.getScore(), None
        
        # set v to -inf
        v = float("-inf")
        # set action to None
        maxAction = None
        
        # # for each action in legal actions
        # for action in state.getLegalActions(agentIndex):
        #     # set v to max of v and min value of successor state
        #     v = max(v, self.minimax(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1)[0])
        
        # # return v and action
        # return v, action
        
        for action in state.getLegalActions(agentIndex):
            newValue = self.minimax(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1)[0]
            if newValue > v:
                v, maxAction = newValue, action
                
        return v, maxAction
    
    def minValue(self, state, depth, agentIndex):
        # if state is terminal, return the state's utility
        if state.isWin() or state.isLose():
            return state.getScore(), None
        
        # set v to inf
        v = float("inf")
        # set action to None
        minAction = None
        
        # for each action in legal actions
        # for action in state.getLegalActions(agentIndex):
        #     # if agent is the last ghost, set v to min of v and max value of successor state
        #     if agentIndex == state.getNumAgents() - 1:
        #         v = min(v, self.minimax(state.generateSuccessor(agentIndex, action), depth - 1, 0)[0])
        #     # else, set v to min of v and min value of successor state
        #     else:
        #         v = min(v, self.minimax(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1)[0])
        
        # # return v and action
        # return v, action
        
        for action in state.getLegalActions(agentIndex):
            # if agent is the last ghost, move to the next depth level and reset agentIndex to 0
            if agentIndex == state.getNumAgents() - 1:
                newValue = self.minimax(state.generateSuccessor(agentIndex, action), depth - 1, 0)[0]
            # else, continue with the next ghost
            else:
                newValue = self.minimax(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1)[0]

            if newValue < v:
                v, minAction = newValue, action

        return v, minAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """

    return currentGameState.getScore()

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

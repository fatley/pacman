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
        newPosition = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]
        

        # checking for each ghost
        # grabbing curr position of pacman & ghost
        oldPosition = currentGameState.getPacmanPosition()
        oldGhostStates = currentGameState.getGhostStates()
        # for each ghost, check if the new position is closer than the old position
        for oldGhostState, newGhostState in zip(oldGhostStates, newGhostStates):
            oldGhostPosition = oldGhostState.getPosition()
            newGhostPosition = newGhostState.getPosition()
            
            # if the new position is closer than the old position, then return -inf
            oldDistance = manhattan(oldPosition, oldGhostPosition)
            newDistance = manhattan(newPosition, newGhostPosition)
            
            if newDistance < 2 and newDistance < oldDistance:
                return float("-inf")
        # capsules
        capsules = currentGameState.getCapsules()
        if capsules:
            closestCapsule = min(capsules, key=lambda capsule: manhattan(newPosition, capsule))
            for ghostState in newGhostStates:
                if manhattan(ghostState.getPosition(), closestCapsule) < 2:
                    foodList = oldFood.asList()
                    if foodList:
                        closestFood = min([manhattan(newPosition, food) for food in foodList])
                        if closestFood == 0:
                            return successorGameState.getScore() + 1
                        else:
                            return successorGameState.getScore() + 1 / closestFood
        
        # if the ghost is scared, eat everythign in sight while the ghost is scared
        if newScaredTimes[0] > 0:
            foodList = oldFood.asList()
            if foodList:
                closestFood = min([manhattan(newPosition, food) for food in foodList])
                if closestFood == 0:
                    return successorGameState.getScore() + 1
                else:
                    return successorGameState.getScore() + 1 / closestFood
            else:
                return successorGameState.getScore() + 1000
        
        # if there are no capsules, carry on with normal routine
        # checking for food
        foodList = oldFood.asList()
        if foodList:
            closestFood = min([manhattan(newPosition, food) for food in foodList])
            if closestFood == 0:
                return successorGameState.getScore() + 1
            else:
                return successorGameState.getScore() + 1 / closestFood
    
        # checking for ghost
        for ghostState in newGhostStates:
            if ghostState.getPosition() < 2:
                return float("-inf")
            
        # if there are no ghost, eat all teh food around
        if not any(ghostState.getPosition() < 3 for ghostState in newGhostStates):
            return successorGameState.getScore() + 1000
        
        return successorGameState.getScore()
    
    
    
    
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
        
    # psuedocode found on - https://www.youtube.com/watch?v=l-hh51ncgDI
    # minimax function (positiionn, depth, maximizing player)
    def minimax(self, state, depth, agentIndex):
        # if depth == 0 or game over in position, return static eval of pos
        if depth == 0 or state.isWin() or state.isLose():
            return self._evaluationFunction(state), None
        
        # if maximizing player (pacman)
        if agentIndex == 0:
            # max evaluation = -inf
            maxEval = float("-inf")
            # for each child of position, eval = minimax(child, depth - 1, false)
            for action in state.getLegalActions(agentIndex):
                # eval = minimax( child, depth -1, false)
                eval = self.minimax(state.generateSuccessor(agentIndex, action), depth - 1, 1)[0]
                # comparing curent action (eval) w / highest action (maxeval) 
                if eval > maxEval:
                    maxEval = max(maxEval, eval)
                    maxAction = action
            return maxEval, maxAction
            
        else:
            # minEval = +inf
            minEval = float("inf")
            # for each child of position, eval = minimax(child, depth - 1, true)
            # minieval = min(mineval, eval)
            for action in state.getLegalActions(agentIndex):
                eval = self.minimax(state.generateSuccessor(agentIndex, action), depth - 1, 0)[0]
                # comparing current action w/ lowest
                if eval < minEval:
                    minEval = min(minEval, eval)
                    minAction = action
            return minEval, minAction
    

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

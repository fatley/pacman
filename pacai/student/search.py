"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue
# from pacai.core.search.problem import SearchProblem


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    # *** Your Code Here ***
    start = problem.startingState()  # start node
    frontier = Stack()
    visited = {start}
    
    # make starting node, if s passes goal, then return
    # frontier - stack
    # visited - make set {start}
    
    # while front isnt empty, pop node recently added to frontier
    # for v = expanded nodes, do if v passes the goal test, then return v
    # if v state not visited, then add v to frontier.
    
    if problem.isGoal(start):
        return start
    
    frontier.push((start, []))
    
    while not frontier.isEmpty():
        # node = frontier.pop() # popping node that was recently added to frontier
        child, actions = frontier.pop()
        # visited.add(node) # adds to visited list
        
        if problem.isGoal(child):
            return actions
        
        for node, action, cost in problem.successorStates(child):
            if node not in visited:
                visited.add(node)
                frontier.push((node, actions + [action]))
        # for child in successors(node):
        # for child in problem.successorStates(node):
            # if child.isGoal():
            # if problem.isGoal(child):
                # return child
            # if child not in visited:
            #     visited.add(child)
            #     frontier.push(child)
    return None

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    start = problem.startingState()  # start node
    
    if problem.isGoal(start):
        return start
    
    frontier = Queue()
    explored = {start}
    frontier.push((start, []))
    
    while not frontier.isEmpty():
        node, actions = frontier.pop()  # chooses the shallowest node in frontier
        explored.add(node)
        
        for child, action, _ in problem.successorStates(node):
            if child not in explored:
                if problem.isGoal(child):
                    return actions + [action]
                frontier.push((child, actions + [action]))
                explored.add(child)
    
    # for child, action, _ in problem.successorStates(node):
        # child = problem.successorStates(node, action)
    # for action in problem.getActions(node):
    #     child = problem.successorStates(node, action)
    #     if child not in explored:
    #         if problem.isGoal(child):
    #             return actions + [action]
    #         frontier.push((child, actions + [action]))
    #         explored.add(child)
    return None

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """
    
    start = problem.startingState()  # start node
    frontier = PriorityQueue()
    explored = set()
    frontier_set = set()
    priorities = {}
    
    frontier.push((start, [], 0), 0)
    frontier_set.add(start)
    priorities[start] = 0
    
    while not frontier.isEmpty():
        node, actions, path_cost = frontier.pop()
        frontier_set.remove(node)
        
        if problem.isGoal(node):
            return actions
    
        explored.add(node)
    
        for child, action, step_cost in problem.successorStates(node):
            total_cost = path_cost + step_cost
            if child not in explored:
                if child not in frontier_set or total_cost < priorities[child]:
                    frontier.push((child, actions + [action], total_cost), total_cost)
                    frontier_set.add(child)
                    priorities[child] = total_cost
    return None

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    
    start = problem.startingState()  # start node
    frontier = PriorityQueue()
    explored = {start}
    frontier_set = {start}
    priorities = {}
    
    frontier.push((start, [], 0), 0)
    frontier_set.add(start)
    priorities[start] = 0
    
    while not frontier.isEmpty():
        node, actions, path_cost = frontier.pop()
        # frontier_set.remove(node)
        if node not in frontier_set:
            continue
        frontier_set.remove(node)
        
        if problem.isGoal(node):
            return actions
        
        explored.add(node)
        
        for child, action, step_cost in problem.successorStates(node):
            total_cost = path_cost + step_cost
            if child not in explored:
                if child not in frontier_set:
                    frontier.push((child, actions + [action], total_cost),
                        total_cost + heuristic(child, problem))
                    frontier_set.add(child)
                    priorities[child] = total_cost + heuristic(child, problem)
                elif total_cost + heuristic(child, problem) < priorities[child]:
                    frontier.push((child, actions + [action], total_cost),
                        total_cost + heuristic(child, problem))
                    priorities[child] = total_cost + heuristic(child, problem)
    return None

    # *** Your Code Here ***
    # raise NotImplementedError()

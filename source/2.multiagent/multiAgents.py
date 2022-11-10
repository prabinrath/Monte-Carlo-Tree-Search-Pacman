# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import numpy as np

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
class MCTS:
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self.max_nodes = 100
        self.node_score = 0
        self.possible_actions = None
        self.possible_actions = self._possible_actions()

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Imports
        from util import manhattanDistance as md

        # Better represented information for convenience
        newFoodList = newFood.asList()
        successorGameScore = successorGameState.getScore()

        numberOfRemainingFood = len(newFoodList)

        distanceFromFoods = [md(newPos, newFoodPos) for newFoodPos in newFoodList]
        distanceFromClosestFood = 0 if (len(distanceFromFoods) == 0) else min(distanceFromFoods)

        distancesFromGhosts = [md(newPos, ngs.getPosition()) for ngs in newGhostStates]
        distanceFromClosestGhost = 0 if (len(distancesFromGhosts) == 0) else min(distancesFromGhosts)
        finalScore = successorGameScore - (1000 if (distanceFromClosestGhost<=1) else (50/distanceFromClosestGhost)) - 50*numberOfRemainingFood - distanceFromClosestFood
        # TODO - This score has to be better. 
        return successorGameScore
        
    def _possible_actions(self):
        if self.state is not None:
            self.possible_actions = self.state.getLegalActions()
        else:
            return None
        return self.possible_actions
    
    def expand(self):
        action = self.possible_actions.pop()
        next_state = self.state.generateSuccessor(0, action)
        child_node = MCTS(
            next_state, parent=self, parent_action=action)
        self.children.append(child_node)
        return child_node 
    
    def rollout(self):
        current_rollout_state = self.state
        if self.state is not None:
            counter  = 0
            while not current_rollout_state.isWin() and not current_rollout_state.isLose():
                possible_moves = current_rollout_state.getLegalActions()
                if counter == self.max_nodes:
                    reward = self.evaluationFunction(current_rollout_state, possible_moves[np.random.randint(len(possible_moves))])
                    return reward
                counter+=1
                if len(possible_moves) > 0:
                    action = possible_moves[np.random.randint(len(possible_moves))]
                else:
                    if current_rollout_state.isWin():
                        return 1000
                    else:
                        return 0
                current_rollout_state = current_rollout_state.generateSuccessor(0, action)
        else:
            return None
        if current_rollout_state.isWin():
            return 1000
        else:
            return 0
            
    def backpropagate(self, result):
        self._number_of_visits += 1
        self.node_score += result
        if self.parent:
            self.parent.backpropagate(result)

    def get_parent_action(self):
        return self.parent_action

    def best_child(self, c_param=2):
        import numpy as np
        weights = [(c.node_score) + c_param * np.sqrt((2 * np.log(self._number_of_visits) / c._number_of_visits)) for c in self.children]
        return self.children[np.argmax(weights)]

    def selection(self):
        current_node = self
        print(str(self.state))
        while len(current_node.state.getLegalActions()) > 0 :
            if not len(current_node.possible_actions) == 0:
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node
    
    def bestAction(self):
        simulation_no = 100
        for i in range(simulation_no):
            v = self.selection()
            reward = v.rollout()
            v.backpropagate(reward)
        return self.best_child()
    
class MonteCarloTreeSearch(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        rootNode = MCTS(state=gameState)
        return rootNode.bestAction().get_parent_action()

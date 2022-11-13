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
import math
from featureExtractors import *
from statistics import mean

from game import Agent

def scoreEvaluationFunction(currentGameState):
    utility = currentGameState.getScore()
    return utility

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

class MCTSNode:
    def __init__(self, state, parent_action=None, ucb_param=2, max_sim_steps = 50):
        self.avg_value = 0
        self.visits = 0
        self.children = []
        self.ucb_param = ucb_param
        self.game_state = state
        self.parent_action = parent_action        
        self.max_sim_steps = max_sim_steps
    
    def iterate(self):
        if self.is_leaf():
            if self.visits == 0:
                # Rollout
                self.avg_value = self.rollout()
                self.visits = 1
                return self.avg_value
            else:
                # Expand
                self.expand()

        # Selection     
        if len(self.children) == 0:
            value = self.rollout()
        else:   
            value = self.children[self.select()].iterate()

        # Backpropagation
        self.avg_value += value
        self.visits += 1   

        return value     

    def rollout(self):
        rollout_state = self.game_state
        num_agents = rollout_state.getNumAgents()
        terminal_flag = False
        sim_itr = 0
        while not terminal_flag:
            for agent_id in range(num_agents):
                sim_itr += 1
                terminal_flag = rollout_state.isWin() or rollout_state.isLose() or sim_itr > self.max_sim_steps
                if terminal_flag:
                    break
                actions = rollout_state.getLegalActions(agent_id)
                rollout_state = rollout_state.generateSuccessor(agent_id, random.choice(actions))

        walls = rollout_state.getWalls()
        value = self.eval_fun(rollout_state, 'food')/(walls.width+walls.height)
        if rollout_state.isWin():
            value += 1
        elif rollout_state.isLose():
            value += -1
        return value

    def expand(self):
        # agent_id 0 for pacman
        for action in self.game_state.getLegalActions(0):
            successor = self.game_state.generateSuccessor(0, action)
            self.children.append(MCTSNode(successor, action, self.ucb_param, self.max_sim_steps))

    def select(self):
        ucb1 = [] # to avoid defining a min ucb1 score
        idx = 0 # to return index directly for rollout
        for child in self.children:
            if child.visits > 0:
                ucb1.append(child.avg_value + self.ucb_param*math.sqrt(math.log(self.visits)/child.visits))
            else:
                return idx
            idx += 1
        return max(range(len(ucb1)), key=lambda i: ucb1[i])

    def is_leaf(self):
        return len(self.children) == 0

    def eval_fun(self, state, arg='ghost'):
        if arg == 'ghost':
            ghost_states = state.getGhostStates()
            pacman_pos = state.getPacmanPosition()
            ghost_proximity = []
            for ghost in ghost_states:
                ghost_proximity.append(manhattanDistance(ghost.getPosition(), pacman_pos)*(1+3*int(ghost.scaredTimer > 0)))
            return min(ghost_proximity)
        elif arg == 'food':
            foods = state.getFood()
            pacman_pos = state.getPacmanPosition()          
            food_proximity = []
            for food in foods.asList():
                food_proximity.append(1/manhattanDistance(food, pacman_pos))
            dist = max(food_proximity) if len(food_proximity) > 0 else 0         
            return dist
    
    def best_action(self):
        # should be called after sufficient iterations
        best_child = self.children[self.select()]
        best_actions = {}
        current_ghost_proximity = self.eval_fun(self.game_state, arg='ghost')
        for child in self.children:
            if abs(child.avg_value - best_child.avg_value) < 1:
                if current_ghost_proximity < 3:
                    best_actions[child.parent_action] = self.eval_fun(child.game_state, arg='ghost')
                else:
                    # best_actions[child.parent_action] = self.eval_fun(child.game_state, arg='food') + child.game_state.getScore()
                    best_actions[child.parent_action] = self.eval_fun(child.game_state, arg='food')
                    if Directions.STOP in best_actions and len(best_actions)>1:
                        del best_actions[Directions.STOP]
        action = max(best_actions, key=best_actions.get)        
        return action
    
class MCTSQAgent(MultiAgentSearchAgent):
    """
      MCTS Agent
    """
    def __init__(self, extractor='SimpleExtractor'):
        self.featExtractor = util.lookup(extractor, globals())()
        MultiAgentSearchAgent.__init__(self)
        self.learn_params = util.Counter()
        self.alpha = 0.2
        self.learning_itr = 0
        self.discount = 0.9

    def temp_print_mct(self,node):  
        print("***** MCTS ********", self.learning_itr)
        print("Parent", node.avg_value, node.visits)
        for child in node.children:
            print(child.avg_value, child.visits, child.parent_action)    
    
    def getQValue(self, state, action):
        features = self.featExtractor.getFeatures(state, action)
        q_val = 0
        for f in features:
            q_val += self.learn_params[f]*features[f]
        return q_val

    def computeValueFromQValues(self, state):
        q_vals = [self.getQValue(state, action) for action in state.getLegalActions()]
        if len(q_vals) > 0:
            return max(q_vals)

        return 0

    def computeActionFromQValues(self, state):
        q_vals = {action: self.getQValue(state, action) for action in state.getLegalActions()}
        if len(q_vals) == 0:
            return None

        if max(q_vals.values()) == 0:            
            better_actions = []
            for k in q_vals:
                if q_vals[k] == 0:
                    better_actions.append(k)
            return random.choice(better_actions)

        return max(q_vals, key=q_vals.get)

    def getAction(self, gameState):
        """
        Returns the best action for the given state using MCTS
        """
        self.learning_itr += 1
        best_action = None
        if self.learning_itr < 500:
            rootNode = MCTSNode(state=gameState)
            n_itr = 10
            while n_itr:
                rootNode.iterate()
                n_itr -= 1
            self.temp_print_mct(rootNode)
            best_action = rootNode.best_action()
            learned_action = self.computeActionFromQValues(gameState)
            print("Learned Action: ", learned_action)
            features = self.featExtractor.getFeatures(gameState, best_action)
            nextState = gameState.generateSuccessor(0, best_action)
            reward = nextState.getScore() - gameState.getScore()
            temporal_diff = (reward+self.discount*self.computeValueFromQValues(nextState)-self.getQValue(gameState, best_action))
            for f in features:
                self.learn_params[f] += self.alpha*temporal_diff*features[f]
        else:    
            best_action = self.computeActionFromQValues(gameState)

        print("Pacman chose: ", best_action)
        return best_action

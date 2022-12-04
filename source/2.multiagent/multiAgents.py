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
from collections import defaultdict, Counter
import json
import copy

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
                    path, _  = closestLoc(self.game_state.getPacmanPosition(), self.game_state.getFood().asList(), self.game_state.getWalls())
                    best_actions[child.parent_action] = int(child.parent_action == path[0]) if path else 0
                    if Directions.STOP in best_actions and len(best_actions)>1:
                        del best_actions[Directions.STOP]
        action = max(best_actions, key=best_actions.get)        
        return action
    
class MonteCarloTreeSearchAgent(MultiAgentSearchAgent):
    """
      MCTS Agent
    """
    def __init__(self, extractor='SimpleExtractor'):
        self.featExtractor = util.lookup(extractor, globals())()
        MultiAgentSearchAgent.__init__(self)
        self.learn_params = defaultdict(Counter)
        self.iter = 0
        self.mode = 'learn'
        # self.mode = 'play'
        try:
            f = open('learn_params.json', 'r')
            temp = json.loads(f.readline())
            for q in temp:
                for d in temp[q]:
                    self.learn_params[q][d] = temp[q][d]
        except:
            pass
            

    def temp_print_mct(self,node):  
        print("***** MCTS ********")
        print("Parent", node.avg_value, node.visits)
        for child in node.children:
            print(child.avg_value, child.visits, child.parent_action)    

    def MCTSAction(self, gameState):
        rootNode = MCTSNode(state=gameState)
        n_itr = 50
        while n_itr:
            rootNode.iterate()
            n_itr -= 1
        self.temp_print_mct(rootNode)
        return rootNode.best_action() 

    def getAction(self, gameState):
        return self.MCTSAction(gameState)

    # def getAction(self, gameState):
    #     """
    #     Returns the best action for the given state using MCTS
    #     """
    #     features = self.featExtractor.getMCTSFeatures(gameState)
    #     if self.mode == 'learn':
    #         self.iter += 1
    #         if self.iter%10 == 0:                
    #             f = open('learn_params.json', 'w')
    #             f.write(json.dumps(self.learn_params))
    #             f.close()
    #             print('Saved MCTS params after ', self.iter, ' iterations.')
    #         best_action = self.MCTSAction(gameState)            
    #         self.learn_params[','.join(features)][best_action] += 1
    #     else:
    #         query = ','.join(features)
    #         if query in self.learn_params:
    #             sample = copy.deepcopy(self.learn_params[query])
    #             best_action = max(sample, key=sample.get)
    #             while not best_action in gameState.getLegalActions():
    #                 del sample[best_action]
    #                 try:
    #                     best_action = max(sample, key=sample.get)
    #                 except:
    #                     print('Sample not Found. Taking a random action.')
    #                     best_action = random.choice(gameState.getLegalActions())
    #         else:
    #             raise Exception('Could not find sample. More training needed.')
    #     # print("Pacman chose: ", best_action)
    #     return best_action

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax(self, gameState, agent_id, depth):
        scores = []
        for action in gameState.getLegalActions(agent_id):
            successor = gameState.generateSuccessor(agent_id, action)
            if depth==1 or successor.isWin() or successor.isLose():
                scores.append((self.evaluationFunction(successor),action))
            else:
                if agent_id == gameState.getNumAgents()-1:
                    agent_id = -1
                scores.append((self.minimax(successor, agent_id+1, depth-1)[0],action))

        if agent_id == 0:
            return max(scores, key=lambda s: s[0])
        else:
            return min(scores, key=lambda s: s[0])

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
        score, action = self.minimax(gameState,0,self.depth*gameState.getNumAgents())
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphabeta(self, gameState, agent_id, depth, alpha, beta):
        if agent_id == 0:
            score = (float('-inf'), None)
        else:
            score = (float('inf'), None)

        for action in gameState.getLegalActions(agent_id):
            successor = gameState.generateSuccessor(agent_id, action)
            if depth==1 or successor.isWin() or successor.isLose():
                if agent_id == 0:
                    score = max([score, (self.evaluationFunction(successor), action)], key=lambda s: s[0])
                    if score[0] > beta:
                        break
                    alpha = max(alpha, score[0])
                else:
                    score = min([score, (self.evaluationFunction(successor), action)], key=lambda s: s[0])
                    if score[0] < alpha:
                        break
                    beta = min(beta, score[0])
            else:
                if agent_id == gameState.getNumAgents()-1:
                    agent_id = -1
                    
                if agent_id == 0:
                    score = max([score, (self.alphabeta(successor, agent_id+1, depth-1, alpha, beta)[0], action)], key=lambda s: s[0])
                    if score[0] > beta:
                        break
                    alpha = max(alpha, score[0])
                else:
                    score = min([score, (self.alphabeta(successor, agent_id+1, depth-1, alpha, beta)[0], action)], key=lambda s: s[0])
                    if score[0] < alpha:
                        break
                    beta = min(beta, score[0])

        return score

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        score, action = self.alphabeta(gameState, 0, self.depth*gameState.getNumAgents(), float('-inf'), float('inf'))
        # print(score, action)
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectimax(self, gameState, agent_id, depth):
        scores = []
        for action in gameState.getLegalActions(agent_id):
            successor = gameState.generateSuccessor(agent_id, action)
            if depth==1 or successor.isWin() or successor.isLose():
                scores.append((self.evaluationFunction(successor),action))
            else:
                if agent_id == gameState.getNumAgents()-1:
                    agent_id = -1
                scores.append((self.expectimax(successor, agent_id+1, depth-1)[0],action))

        if agent_id == 0:
            return max(scores, key=lambda s: s[0])
        else:
            values = [score[0] for score in scores] 
            return (sum(values)/len(values), None)

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        score, action = self.expectimax(gameState,0,self.depth*gameState.getNumAgents())
        return action
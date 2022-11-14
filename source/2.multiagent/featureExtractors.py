# featureExtractors.py
# --------------------
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


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestLoc(pos, obj, walls):
    """
    closestLoc -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], [], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, path, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if (pos_x,pos_y) in obj:
            return path, dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            dx = nbr_x-pos_x
            dy = nbr_y-pos_y
            dir = None
            if dx == 1:
                dir = Directions.EAST
            if dx == -1:
                dir = Directions.WEST
            if dy == 1:
                dir = Directions.NORTH
            if dy == -1:
                dir = Directions.SOUTH
            if dx == 0 and dy == 0:
                dir = Directions.STOP
            fringe.append((nbr_x, nbr_y, path+[dir], dist+1))
    # no food found
    return None, None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostStates()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        # features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
        features["#-of-ghosts-1-step-away"] = sum(((next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls) and g.scaredTimer == 0) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        _, dist = closestLoc((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features
    
    def loc_image(self, loc, foods, ghosts, walls):
        # if foods[loc[0]][loc[1]]:
        #     return 'F'
        # if walls[loc[0]][loc[1]]:
        #     return 'W'
        for g in ghosts:
            if loc == g.getPosition() and g.scaredTimer == 0:
                return 'G'
        return 'N'

    def getMCTSFeatures(self, state):
        features = []
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostStates()
        
        px, py = state.getPacmanPosition()
        features.append(self.loc_image((px-1,py), food, ghosts, walls))
        features.append(self.loc_image((px,py-1), food, ghosts, walls))
        features.append(self.loc_image((px+1,py), food, ghosts, walls))
        features.append(self.loc_image((px,py+1), food, ghosts, walls))
        
        path_food, _ = closestLoc(state.getPacmanPosition(), food.asList(), walls)        
        if path_food:
            features.append(path_food[0])
        else:
            features.append('None')
        # ghosts_locs = []
        # for g in ghosts:
        #     if g.scaredTimer == 0:
        #         ghosts_locs.append(g.getPosition())
        # path_ghost, dist_ghost = closestLoc(state.getPacmanPosition(), ghosts_locs, walls)
        # if dist_ghost:
        #     features.append(path_ghost[0]) if dist_ghost < 3 else features.append('None')
        # else:
        #     features.append('None')
        
        return features

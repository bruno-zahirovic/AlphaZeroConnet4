import numpy as np
import math

import torch

import torch.nn as nn
import torch.nn.functional as F

class Node:
    def __init__(self, game, args, state, parent=None, actionTaken=None, prior=0, visitCount=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.actionTaken = actionTaken
        self.prior = prior

        self.children = []

        self.visitCount = visitCount
        self.valueSum = 0

    def IsFullyExpanded(self):
        return len(self.children) > 0
    
    def Select(self):
        bestChild = None
        bestUcb = -np.inf

        for child in self.children:
            ucb = self.GetUcb(child)
            if ucb > bestUcb:
                bestChild = child
                bestUcb = ucb

        return bestChild
    
    def GetUcb(self, child):
        if child.visitCount == 0:
            qVal = 0
        else:
            qVal = 1 - ((child.valueSum / child.visitCount) + 1) / 2
        return qVal + self.args['C'] * math.sqrt(self.visitCount / (child.visitCount + 1)) * child.prior
    
    def Expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                childState = self.state.copy()
                childState = self.game.GetNextState(childState, action, 1)
                childState = self.game.ChangePerspective(childState, player=-1)

                child = Node(self.game, self.args, childState, self, action, prob)
                self.children.append(child)
    
    
    def BackPropagate(self, value):
        self.valueSum += value
        self.visitCount += 1

        value = self.game.GetOpponentValue(value)

        if self.parent is not None:
            self.parent.BackPropagate(value)
    
"""     def Simulate(self): #Not necessary for AlphaZero mcts
        value, isTerminal = self.game.GetValueAndTerminated(self.state, self.actionTaken)
        value = self.game.GetOpponentValue(value)

        if isTerminal:
            return value
        
        rolloutState = self.state.copy()
        rolloutPlayer = 1

        while True:
            validMoves = self.game.GetValidMoves(rolloutState)
            action = np.random.choice(np.where(validMoves == 1)[0])
            rolloutState = self.game.GetNextState(rolloutState, action, rolloutPlayer)

            value, isTerminal = self.game.GetValueAndTerminated(rolloutState, action)
            if isTerminal:
                if rolloutPlayer == -1:
                    value = self.game.GetOpponentValue(value)
                return value
            
            rolloutPlayer = self.game.GetOpponent(rolloutPlayer) """



    



class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def Search(self, state):
        # Define Root Node
        root = Node(self.game, self.args, state, visitCount=1)

        policy, _ = self.model(
            torch.tensor(self.game.GetEncodedState(state), device=self.model.device).unsqueeze(0)
        )

        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        
        # Add noise to policy to improve exploration
        policy = (1 - self.args['dirichlet_epsilon']) * policy + \
        (self.args['dirichlet_epsilon'] * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.actionSize))

        validMoves = self.game.GetValidMoves(state)
        policy *= validMoves

        policy /= np.sum(policy)
        root.Expand(policy)

        for search in range(self.args['num_searches']):
            node = root

            #Selection
            while node.IsFullyExpanded():
                node = node.Select()

            value, isTerminal = self.game.GetValueAndTerminated(node.state, node.actionTaken)
            value = self.game.GetOpponentValue(value)

            if not isTerminal:
                
                policy, value = self.model(
                    torch.tensor(self.game.GetEncodedState(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy() #squeeze to remove batching
                validMoves =  self.game.GetValidMoves(node.state)
                policy *= validMoves
                policy /= np.sum(policy)

                value = value.item()

                #Expansion
                node.Expand(policy)
                #Simulation
                #value = node.Simulate() #Remove simulation because AlphaZero mcts implementation doesn't need it

            #BackPropagation
            node.BackPropagate(value)

        #Return visitCounts
        actionProbs = np.zeros(self.game.actionSize)
        for child in root.children:
            actionProbs[child.actionTaken] = child.visitCount
        actionProbs /= np.sum(actionProbs)
        return actionProbs
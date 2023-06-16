import numpy as np
import math

class Node:
    def __init__(self, game, args, state, parent=None, actionTaken=None):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.actionTaken = actionTaken

        self.children = []
        self.expandableMoves = game.GetValidMoves(state)

        self.visitCount = 0
        self.valueSum = 0

    def IsFullyExpanded(self):
        return np.sum(self.expandableMoves) == 0 and len(self.children) > 0
    
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
        qVal = 1 - ((child.valueSum / child.visitCount) + 1) / 2
        return qVal + self.args['C'] * math.sqrt(math.log(self.visitCount) / child.visitCount)
    
    def Expand(self):
        action = np.random.choice(np.where(self.expandableMoves == 1)[0])
        self.expandableMoves[action] = 0

        childState = self.state.copy()
        childState = self.game.GetNextState(childState, action, 1)
        childState = self.game.ChangePerspective(childState, player=-1)

        child = Node(self.game, self.args, childState, self, action)
        self.children.append(child)
        return child
    
    def Simulate(self):
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
            
            rolloutPlayer = self.game.GetOpponent(rolloutPlayer)

    def BackPropagate(self, value):
        self.valueSum += value
        self.visitCount += 1

        value = self.game.GetOpponentValue(value)

        if self.parent is not None:
            self.parent.BackPropagate(value)


    



class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args

    def Search(self, state):
        # Define Root Node
        root = Node(self.game, self.args, state)

        for search in range(self.args['num_searches']):
            node = root

            #Selection
            while node.IsFullyExpanded():
                node = node.Select()

            value, isTerminal = self.game.GetValueAndTerminated(node.state, node.actionTaken)
            value = self.game.GetOpponentValue(value)

            if not isTerminal:
                #Expansion
                node = node.Expand()

                #Simulation
                value = node.Simulate()

            #BackPropagation
            node.BackPropagate(value)

        #Return visitCounts
        actionProbs = np.zeros(self.game.actionSize)
        for child in root.children:
            actionProbs[child.actionTaken] = child.visitCount
        actionProbs /= np.sum(actionProbs)
        return actionProbs
from model import Connect4Model
import numpy as np
import os
import math
from connect4 import Connect4
import copy
import collections

ALPHA = 0.001
EPSILON = 0.8
BATCH_SIZE = 32
EPOCH_NUM = 50
MAX_ITERATIONS = 500
MCTS_GAMES = 80
EVAL_GAMES = 100

class Node:
    def __init__(self, game, move, parent=None):
        self.game = game # state s
        self.move = move # action index
        self.isExpanded = False
        self.parent = parent  
        self.children = {}
        self.childPriors = np.zeros([7], dtype=np.float32)
        self.childTotalValue = np.zeros([7], dtype=np.float32)
        self.childNumberVisited = np.zeros([7], dtype=np.float32)
        self.actionIndexes = []

    @property
    def numberVisits(self):
        return self.parent.childNumberVisits[self.move]

    @numberVisits.setter
    def numberVisits(self, value):
        self.parent.childNumberVisits[self.move] = value
    
    @property
    def totalValue(self):
        return self.parent.childTotalValue[self.move]
    
    @totalValue.setter
    def totalValue(self, value):
        self.parent.childTotalValue[self.move] = value

    def childQValue(self):
        return self.childTotalValue / (1 + self.childNumberVisited)

    def childUValue(self):
        return math.sqrt(self.numberVisits) * (abs(self.childPriors) / (1 + self.childNumberVisited))
    
    def bestChild(self):
        if self.actionIndexes != []:
            bestMove = self.childQValue() + self.childUValue()
            bestMove = self.actionIndexes[np.argmax(bestMove[self.actionIndexes])]
        else:
            bestMove = np.argmax(self.childQValue() + self.childUValue())
        return bestMove

    def selectLeaf(self):
        current = self
        while current.isExpanded:
            bestMove = current.bestChild()
            current = current.possiblyAddChild(bestMove)
        return current

    def addDirichletNoise(self, actionIndexes, childPriors):
        validChildPriors = childPriors[actionIndexes]
        validChildPriors = 0.75 * validChildPriors + 0.25 * np.random.dirichlet(np.zeros([len(validChildPriors)], dtype=np.float32) + 192)
        
        childPriors[actionIndexes] = validChildPriors
        return childPriors

    def expand(self, childPriors):
        self.isExpanded = True
        actionIndexes = self.game.validActions()
        priors = childPriors

        if actionIndexes == []:
            self.isExpanded = False

        self.actionIndexes = actionIndexes
        priors[[i for i in range(len(childPriors)) if i not in actionIndexes]] = 0.0

        if self.parent.parent == None:
            priors = self.addDirichletNoise(actionIndexes, priors)

        self.childPriors = priors


    def decodePiecesNMoves(self, board, move):
        board.updateBoard(move)
        return board

    def possiblyAddChild(self, move):
        if move not in self.children:
            boardCopy = copy.deepcopy(self.game)
            boardCopy = self.decodePiecesNMoves(boardCopy, move)

            self.children[move] = Node(boardCopy, move, parent=self)
            
            return self.children[move]

    def backup(self, valueEstimate:float):
        current = self
        
        while current.parent is not None:
            current.numberVisits += 1
            if current.game.turn == -1:
                current.totalValue += (-1*valueEstimate)
            elif current.game.turn == 1:
                current.totalValue += (valueEstimate)

            current = current.parent


class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.childTotalValue = collections.defaultdict(float)
        self.childNumberVisists = collections.defaultdict(float)


def UCTSearch(gameState):
    pass

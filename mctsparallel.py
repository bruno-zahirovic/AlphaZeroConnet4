import numpy as np
import math

import torch

import torch.nn as nn
import torch.nn.functional as F

from mcts_new import Node

class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def Search(self, states, selfPlayGames):
        # Define Root Node
        
        policy, _ = self.model(
            torch.tensor(self.game.GetEncodedState(states), device=self.model.device)
        )

        policy = torch.softmax(policy, axis=1).cpu().numpy()    
        
        # Add noise to policy to improve exploration
        policy = ((1 - self.args['dirichlet_epsilon']) * policy) + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.actionSize, size=policy.shape[0])

        for i, spg in enumerate(selfPlayGames):
            spgPolicy = policy[i]
            validMoves = self.game.GetValidMoves(states[i])
            spgPolicy *= validMoves
            #spgPolicy /= np.sum(spgPolicy)
            spg.root = Node(self.game, self.args, states[i], visitCount=1)
            spg.root.Expand(spgPolicy)

        for search in range(self.args['num_searches']):
            for spg in selfPlayGames:
                spg.node = None
                node = spg.root

                #Selection
                while node.IsFullyExpanded():
                    node = node.Select()

                value, isTerminal = self.game.GetValueAndTerminated(node.state, node.actionTaken)
                value = self.game.GetOpponentValue(value)

                if isTerminal:
                    #BackPropagation
                    node.BackPropagate(value)

                else:
                    spg.node = node


            expandableSelfPlayGames = [mappingIdx for mappingIdx in range(len(selfPlayGames)) if selfPlayGames[mappingIdx].node is not None]

            if len(expandableSelfPlayGames) > 0:
                states = np.stack([selfPlayGames[mappingIdx].node.state for mappingIdx in expandableSelfPlayGames])
                policy, value = self.model(
                    torch.tensor(self.game.GetEncodedState(states), device=self.model.device)
                )
                policy = torch.softmax(policy, axis=1).cpu().numpy()
                value = value.cpu().numpy()

            for i, mappingIdx in enumerate(expandableSelfPlayGames):
                node = selfPlayGames[mappingIdx].node
                spgPolicy, spgValue = policy[i], value[i]

                validMoves =  self.game.GetValidMoves(node.state)
                spgPolicy *= validMoves
                spgPolicy /= np.sum(spgPolicy)

                #Expansion
                node.Expand(spgPolicy)
                node.BackPropagate(spgValue)
                #Simulation
                #value = node.Simulate() #Remove simulation because AlphaZero mcts implementation doesn't need it
                
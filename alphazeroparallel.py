import torch
import numpy as np
from tqdm.notebook import tnrange
import random

from torch.nn import functional as F

from mctsparallel import MCTSParallel
from tictactoe import TicTacToe
from model_new import ResNet
from connect4 import Connect4


class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args

        self.mcts = MCTSParallel(game, args, model)

    def SelfPlay(self):
        returnMemory = []
        player = 1
        selfPlayGames = [SelfPlayGame(self.game) for spg in range(self.args['num_parallel_games'])]

        while len(selfPlayGames) > 0:
            states = np.stack([spg.state for spg in selfPlayGames])

            neutralStates = self.game.ChangePerspective(states, player)
            self.mcts.Search(neutralStates, selfPlayGames)

            for i in range(len(selfPlayGames))[::-1]:
                spg = selfPlayGames[i]

                actionProbs = np.zeros(self.game.actionSize)
                for child in spg.root.children:
                    actionProbs[child.actionTaken] = child.visitCount
                actionProbs /= np.sum(actionProbs)

                spg.memory.append((spg.root.state, actionProbs, player))

                temperatureActionProbs = actionProbs ** (1 / self.args['temperature']) #exploitation/exploration
                temperatureActionProbs /= np.sum(temperatureActionProbs)

                action = np.random.choice(self.game.actionSize, p=temperatureActionProbs)
                spg.state = self.game.GetNextState(spg.state, action, player)

                value, isTerminal = self.game.GetValueAndTerminated(spg.state, action)

                if isTerminal:
                    for histNeutralState, histActionProbs, histPlayer in spg.memory:
                        histOutcome = value if histPlayer == player else self.game.GetOpponentValue(value)
                        returnMemory.append((
                            self.game.GetEncodedState(histNeutralState), 
                            histActionProbs,
                            histOutcome
                        ))
                    del selfPlayGames[i]

            player = self.game.GetOpponent(player)
        return returnMemory
    
    def Train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])]

            state, policyTargets, valueTargets = zip(*sample)

            state, policyTargets, valueTargets = np.array(state), np.array(policyTargets), np.array(valueTargets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policyTargets = torch.tensor(policyTargets, dtype=torch.float32, device=self.model.device)
            valueTargets = torch.tensor(valueTargets, dtype=torch.float32, device=self.model.device)

            outPolicy, outValue = self.model(state)

            policyLoss = F.cross_entropy(outPolicy, policyTargets)
            valueLoss = F.mse_loss(outValue, valueTargets)

            loss = policyLoss + valueLoss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()



    def Learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []

            self.model.eval()
            for selfPlayIter in tnrange(self.args['num_self_play_iterations'] // self.args["num_parallel_games"]):
                memory += self.SelfPlay()

            self.model.train()
            for epoch in tnrange(self.args['num_epochs']):
                self.Train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{self.game}.pt")


class SelfPlayGame:
    def __init__(self, game):
        self.state = game.GetInitialState()
        self.memory = []
        self.root = None
        self.node = None


def main():
    gameInstance = Connect4()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(gameInstance, 9, 128, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    args = {
        'C': 2,
        'num_searches': 600,
        'num_iterations': 8,
        'num_self_play_iterations': 500,
        'num_parallel_games': 100,
        'num_epochs': 4,
        'batch_size': 128,
        "temperature": 1.25,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 0.3
    }

    alphaZeroParallel = AlphaZeroParallel(model, optimizer, gameInstance, args)
    alphaZeroParallel.Learn()

if __name__ == "__main__":
    main()



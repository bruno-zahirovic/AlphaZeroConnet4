import torch
import numpy as np
from tqdm.notebook import tnrange
import random

from torch.nn import functional as F

from mcts_new import MCTS
from tictactoe import TicTacToe
from model_new import ResNet

class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args

        self.mcts = MCTS(self.game, self.args, self.model)

    def SelfPlay(self):
        memory = []
        player = 1
        state = self.game.GetInitialState()

        while True:
            neutralState = self.game.ChangePerspective(state, player)
            actionProbs = self.mcts.Search(neutralState)

            memory.append((neutralState, actionProbs, player))

            temperatureActionProbs = actionProbs ** (1 / self.args["temperature"]) #exploitation/exploration
            temperatureActionProbs /= np.sum(temperatureActionProbs)

            action = np.random.choice(self.game.actionSize, p=temperatureActionProbs)
            state = self.game.GetNextState(state, action, player)

            value, isTerminal = self.game.GetValueAndTerminated(state, action)

            if isTerminal:
                returnMemory = []
                for histNeutralState, histActionProbs, histPlayer in memory:
                    histOutcome = value if histPlayer == player else self.game.GetOpponentValue(value)
                    returnMemory.append((
                        self.game.GetEncodedState(histNeutralState), 
                        histActionProbs,
                        histOutcome
                    ))
                return returnMemory
            player = self.game.GetOpponent(player)
    
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
            for selfPlayIter in tnrange(self.args['num_self_play_iterations']):
                memory += self.SelfPlay()

            self.model.train()
            for epoch in tnrange(self.args['num_epochs']):
                self.Train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")


def main():
    gameInstance = TicTacToe()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(gameInstance, 4, 64, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    args = {
        'C': 2,
        'num_searches': 60,
        'num_iterations': 3,
        'num_self_play_iterations': 500,
        'num_epochs': 4,
        'batch_size': 64,
        "temperature": 1.25,
        "dirichlet_epsilon": 0.25,
        "dirichlet_alpha": 0.3
    }

    alphaZero = AlphaZero(model, optimizer, gameInstance, args)
    alphaZero.Learn()

if __name__ == '__main__':
    main()


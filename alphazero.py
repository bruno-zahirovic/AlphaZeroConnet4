import torch
import numpy as np
from tqdm.notebook import trange

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

            action = np.random.choice(self.game.actionSize, p=actionProbs)
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
        pass

    def Learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []

            self.model.eval()
            for selfPlayIter in trange(self.args['num_self_play_iterations']):
                memory += self.SelfPlay()

            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.Train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")


def main():
    gameInstance = TicTacToe()
    model = ResNet(gameInstance, 4, 64)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    args = {
        'C': 2,
        'num_searches': 60,
        'num_iterations': 3,
        'num_self_play_iterations': 10,
        'num_epochs': 4
    }

    alphaZero = AlphaZero(model, optimizer, gameInstance, args)
    alphaZero.Learn()

if __name__ == '__main__':
    main()


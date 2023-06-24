from mcts_new import MCTS
from model_new import ResNet
from tictactoe import TicTacToe
from connect4 import Connect4
from alphazero import AlphaZero

import torch
import numpy as np


def main():
    gameInstance = Connect4()
    player = 1

    args = {
        "C": 2,
        "num_searches": 100,
        "temperature": 1.25,
        "dirichlet_epsilon": 0.0,
        "dirichlet_alpha": 0.3
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(gameInstance, 9, 128, device=device)
    model.eval()

    mcts = MCTS(gameInstance, args, model=model)
    state = gameInstance.GetInitialState()

    while True:
        print(state)

        if player == 1:
            validMoves = gameInstance.GetValidMoves(state)
            print("Valid Moves", [i for i in range(gameInstance.actionSize) if validMoves[i] == 1])
            action = int(input(f"{player}:"))

            if validMoves[action] == 0:
                print("Action not Valid!")
                continue
        else:
            neutralState = gameInstance.ChangePerspective(state, player)
            mctsProbs = mcts.Search(neutralState)

            action = np.argmax(mctsProbs)


        state = gameInstance.GetNextState(state, action, player)

        value, isTerminal = gameInstance.GetValueAndTerminated(state, action)

        if isTerminal:
            print(state)
            if value == 1:
                print(player, "Won!")
            else:
                print("Draw!")

            break
        player = gameInstance.GetOpponent(player)


if __name__ == '__main__':
    main()
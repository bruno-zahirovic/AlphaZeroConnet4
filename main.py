from mcts_new import MCTS
from model_new import ResNet
from tictactoe import TicTacToe
from connect4 import Connect4
from alphazero import AlphaZero

import torch
import numpy as np

import pygame


def main():
    gameInstance = Connect4(displayActive=True)

    args = {
        "C": 2,
        "num_searches": 600,
        "dirichlet_epsilon": 0.0,
        "dirichlet_alpha": 0.3
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet(gameInstance, 9, 128, device=device)
    model.load_state_dict(torch.load('model_10_Connect4.pt', map_location=device))
    model.eval()

    mcts = MCTS(gameInstance, args, model=model)

    """ while True:
        print(state)

        if player == 1:
            gameInstance.DropPiece()
            pygame.display.update()
            validMoves = gameInstance.GetValidMoves(state)
            print("Valid Moves", [i for i in range(gameInstance.actionSize) if validMoves[i] == 1])
            #print("Encoded:", gameInstance.GetEncodedState(state))
            #action = int(input(f"{player}:"))
            action = gameInstance.selection

            if validMoves[action] == 0:
                print("Action not Valid!")
                continue
        else:
            neutralState = gameInstance.ChangePerspective(state, player)
            mctsProbs = mcts.Search(neutralState)

            action = np.argmax(mctsProbs)


        state = gameInstance.GetNextState(state, action, player)
        gameInstance.board=state
        gameInstance.UpdateBoard(action)

        value, isTerminal = gameInstance.GetValueAndTerminated(state, action)

        if isTerminal:
            print(state)
            if value == 1:
                print(player, "Won!")
            else:
                print("Draw!")

            break
        player = gameInstance.GetOpponent(player) """
    
    while (True):
        gameInstance = Connect4(displayActive=True)
        gameInstance.turn = 1
        while not gameInstance.gameOver:
            if gameInstance.turn == 1:
                gameInstance.DropPiece()
            else:
                neutralState = gameInstance.ChangePerspective(gameInstance.board, gameInstance.turn)
                mctsProbs = mcts.Search(neutralState)
                action = np.argmax(mctsProbs)

                #gameInstance.board = gameInstance.GetNextState(gameInstance.board, action, gameInstance.turn)
                gameInstance.UpdateBoard(action)
                gameInstance.PrintBoard()

                isTerminated = gameInstance.CheckForGameOver()
                if isTerminated:
                    gameInstance.HandleWin()
                    gameInstance.gameOver = True
                gameInstance.turn = gameInstance.GetOpponent(gameInstance.turn)
            pygame.display.update()
        pygame.time.wait(3000)
        pygame.quit()
        


if __name__ == '__main__':
    main()
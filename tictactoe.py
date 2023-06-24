import numpy as np
import sys, os

from mcts_new import MCTS
#from model_new import ResNet

class TicTacToe:
    def __init__(self):
        self.rowCount = 3
        self.colCount = 3
        self.actionSize = self.rowCount * self.colCount

    def __repr__(self):
        return "TicTacToe"

    def GetInitialState(self):
        return np.zeros((self.rowCount, self.colCount))
    
    def GetNextState(self, state, action, player):
        row = action // self.colCount
        col = action % self.colCount
        state[row, col] = player
        return state
    
    def GetValidMoves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def CheckWin(self, state, action):
        if action == None:
            return False

        row = action // self.colCount
        col = action % self.colCount
        player =  state[row, col]

        return (
            np.sum(state[row, :]) == player * self.colCount 
            or np.sum(state[:, col]) == player * self.rowCount
            or np.sum(np.diag(state)) == player * self.rowCount
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.rowCount
        )
    
    def GetValueAndTerminated(self,state, action):
        if self.CheckWin(state, action):
            return 1, True
        if np.sum(self.GetValidMoves(state)) == 0:
            return 0, True
        return 0, False
    
    def GetOpponent(self, player):
        return -player
    
    def GetOpponentValue(self, value):
        return -value
    
    def ChangePerspective(self, state, player):
        return state * player
    
    def GetEncodedState(self, state):
        encodedState = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        return encodedState



""" def main():
    gameInstance = TicTacToe()
    player = 1

    args = {
        "C": 2,  #sqrt(2)
        "num_searches": 1000
    }

    model = ResNet(gameInstance, 4, 64)
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


if __name__ == "__main__":
    main() """
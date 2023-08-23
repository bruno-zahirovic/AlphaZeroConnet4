
from sqlite3 import SQLITE_ALTER_TABLE
from turtle import window_height
import numpy as np
import pygame
from scipy.signal import convolve2d
import sys
import time

AI_PLAY = True

class Connect4():
    def __init__(self, displayActive=False):
        self.gameOver = False
        self.isDraw = False
        self.turn = -1
        self.rowCount = 6
        self.colCount = 7
        self.actionSize = self.colCount
        self.winCount = 4
        self.squareSize = 100
        self.selection = 0
        self.pieceRadius = int((self.squareSize / 2) - 5)
        self.windowWidth = (self.colCount + 1) * self.squareSize 
        self.windowHeight = (self.rowCount + 2) * self.squareSize
        self.winKernels = [np.array([[1, 1, 1, 1]]), \
                        np.transpose(np.array([[1, 1, 1, 1]])), \
                        np.eye(4, dtype=np.uint8), \
                        np.fliplr(np.eye(4, dtype=np.uint8))]

        self.colors = { "blue": (47, 110, 240), \
                        "red": (204, 20, 29), \
                        "green": (0, 255, 0), \
                        "yellow": (255, 255, 117), \
                        "black": (23, 23, 23), \
                        "white": (245, 243, 245), \
                        "grey100": (100, 100, 100), \
                        "grey": (29, 29, 27), \
                        "cyan": (0,255,255)}
        self.bgColor = self.colors["grey"]
        self.board = np.zeros((self.rowCount, self.colCount))
        if displayActive:
            pygame.init()
            self.screen = pygame.display.set_mode((self.windowWidth, self.windowHeight))
            self.screen.fill(self.bgColor)
            pygame.display.set_caption("Connect4")
            self.DrawBoard()
            pygame.display.update()

    def __repr__(self):
        return "Connect4"
    
    def DrawBoard(self):
        for col in range(self.colCount):
            for row in range(self.rowCount):
                pygame.draw.rect(self.screen, self.colors["blue"], (col * self.squareSize + (self.squareSize / 2), (row + 1) * self.squareSize + (self.squareSize / 2), self.squareSize, self.squareSize))
                pygame.draw.circle(self.screen, self.bgColor, ((col + 1) * self.squareSize, (row + 2) * self.squareSize), self.pieceRadius)
        pygame.draw.rect(self.screen, self.colors["blue"], ((self.colCount * self.squareSize) + int(self.squareSize / 2.5), ((self.rowCount + 1) * self.squareSize + (self.squareSize / 2)), int(self.squareSize / 3), int(self.squareSize / 2)))
        pygame.draw.rect(self.screen, self.colors["blue"], ((self.squareSize / 2) - (int(self.squareSize / 4)), ((self.rowCount + 1) * self.squareSize + (self.squareSize / 2)), int(self.squareSize / 3), int(self.squareSize / 2)))

    def DropPiece(self):
        if self.turn == 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                elif event.type == pygame.MOUSEMOTION:
                    self.__handleOnMouseMotionEvent(event)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.__handleOnMouseClickEvent(event)
        

    def __handleOnMouseMotionEvent(self, event):
        pygame.draw.rect(self.screen, self.bgColor, (0, 0, self.windowWidth + (self.squareSize / 2), self.squareSize + (self.squareSize / 2)))
        if self.turn == 1:
            pygame.draw.circle(self.screen, self.colors["red"], (event.pos[0], int(self.squareSize / 1.2)), self.pieceRadius)
        else:
            pygame.draw.circle(self.screen, self.colors["yellow"], (event.pos[0], int(self.squareSize / 1.2)), self.pieceRadius)
        self.__drawCurrentPlayerString()

    def __handleOnMouseClickEvent(self, event):
        pygame.draw.rect(self.screen, self.bgColor, (0, 0, self.windowWidth + (self.squareSize / 2), self.squareSize + (self.squareSize / 2)))
        self.__playStep(event)
        if self.turn == 1:
            pygame.draw.circle(self.screen, self.colors["red"], (event.pos[0], int(self.squareSize / 1.2)), self.pieceRadius)
        else:
            pygame.draw.circle(self.screen, self.colors["yellow"], (event.pos[0], int(self.squareSize / 1.2)), self.pieceRadius)
        self.__drawCurrentPlayerString()
    
    def __drawCurrentPlayerString(self):
        myFont = pygame.font.SysFont("monospace", 25, bold=True)
        currentPlayer, color = self.__handleCurrentPlayerString()
        label = myFont.render(currentPlayer, 1, color)
        self.screen.blit(label, (5, 5))

    def __playStep(self, event):
        self.__handleSelection(event)
        if self.selection != -1:
            self.__updateBoard()
            self.__printBoard()
            self.CheckForGameOver()
            if self.gameOver == True:
                self.__handleGameOver()
            self.__handleTurn()
        else:
            if self.__isDraw():
                self.CheckForGameOver()
            print("INVALID MOVE, CHOOSE AGAIN!")
        pygame.display.update()

    def __handleSelection(self, event):
        self.selection = int(event.pos[0] / (self.squareSize) + 0.5) - 1
        self.selection = self.__clampSelection(self.selection, 0, self.colCount - 1)
        if not self.__isValidSelection():
            self.selection = -1

    def __clampSelection(self, val, minVal, maxVal):
        return max(min(maxVal, val), minVal)
    
    def __isValidSelection(self):
        if (self.selection < 0 or self.selection >= self.colCount) or (self.board[0][self.selection]) != 0:
            return False
        return True

    def __updateBoard(self):
        for i in range(self.rowCount):
            if self.board[self.rowCount - 1 - i][self.selection] != 0:
                continue
            self.board[self.rowCount - 1 - i][self.selection] = self.turn
            if self.turn == 1:
                color = self.colors["red"]
            else:
                color = self.colors["yellow"]
            pygame.draw.circle(self.screen, color, ((self.squareSize * (1 + self.selection)),self.squareSize * (1 + (self.rowCount-i))), self.pieceRadius)
            break

    def __printBoard(self):
        print(np.flip(self.board, 0))

    def CheckForGameOver(self):
        for kernel in self.winKernels:
            if (convolve2d(self.board == self.turn, kernel, mode = "valid") == 4).any():
                self.gameOver = True

        if self.__isDraw():
            self.isDraw = True
            self.gameOver = True
        return self.gameOver

    def __handleTurn(self):
        if self.turn == -1:
            self.turn = 1
        else:
            self.turn = -1

    def __handleGameOver(self):
        if self.isDraw:
            self.__handleDraw()
        else:
            self.HandleWin()
        pygame.display.update()

    def __isDraw(self):
        if self.ValidActions() == [] and self.gameOver == False:
            return True
        return False

    def __handleDraw(self):
        print("***GAME OVER***")
        print("IT'S A DRAW!!")
        myFont = pygame.font.SysFont("monospace", 50, bold=True)
        label = myFont.render("IT'S A DRAW!!", 1, self.colors["white"])
        pygame.draw.rect(self.screen, self.bgColor, (205, 250, 400, 50))
        self.screen.blit(label, (210, 250))

    def HandleWin(self):
        winner, color = self.__handleWinnerString()
        print("***GAME OVER***")
        print("WINNER: ", winner)
        myFont = pygame.font.SysFont("monospace", 50, bold=True)
        label = myFont.render("WINNER: " + winner + "!!", 1, color)
        pygame.draw.rect(self.screen, self.bgColor, (130, 250, 540, 55))
        self.screen.blit(label, (135, 250))

    def __handleWinnerString(self):
        if self.turn == 1:
            playerString = "PLAYER 1"
            color = self.colors["red"]
        else:
            if not AI_PLAY:
                playerString ="PLAYER 2"
            else:
                playerString ="AlphaZero"
            color = self.colors["yellow"]
        return playerString, color
    
    def __handleCurrentPlayerString(self):
        if self.turn == 1:
            playerString = "PLAYER 1"
            color = self.colors["red"]
        else:
            if not AI_PLAY:
                playerString = "PLAYER 2"
            else:
                playerString = "AlphaZero is thinking.."
            color = self.colors["yellow"]
        return playerString, color

    def ValidActions(self):
        actions = []
        for col in range(self.colCount):
            if self.board[0][col] == 0:
                actions.append(col)
        return actions

    def UpdateBoard(self, move):
        ###Assume the move selected is valid
        for i in range(self.rowCount):
            if self.board[self.rowCount - 1 - i][move] != 0:
                continue
            self.board[self.rowCount - 1 - i][move]  = self.turn
            if self.turn == 1:
                color = self.colors["red"]
            else:
                color = self.colors["yellow"]
            pygame.draw.circle(self.screen, color, ((self.squareSize * (1 + move)),self.squareSize * (1 + (self.rowCount-i))), self.pieceRadius)
            pygame.display.update()
            break

    def GetInitialState(self):
        return np.zeros((self.rowCount, self.colCount))
    
    def GetNextState(self, state, action, player):
        row = np.max(np.where(state[:, action] == 0))
        col = action    
        state[row][col] = player
        self.board = state
        return state

    def GetValidMoves(self, state):
        return (state[0] == 0).astype(np.uint8)
    
    def CheckWin(self, state, action):
        if action == None:
            return False
        
        row = np.min(np.where(state[:, action] != 0))
        column = action
        player = state[row][column]
        
        for kernel in self.winKernels:
            if (convolve2d(state == player, kernel, mode = "valid") == self.winCount).any():
                return True
        return False
    
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

        if len(state.shape) == 3:
            encodedState = np.swapaxes(encodedState, 0, 1)

        return encodedState
    


if __name__ == "__main__":
    while (True):
        gameInstance = Connect4(displayActive=True)
        while not gameInstance.gameOver:
            gameInstance.DropPiece()
            pygame.display.update()
        pygame.time.wait(3000)
        pygame.quit()

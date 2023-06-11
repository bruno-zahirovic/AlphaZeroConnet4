
from sqlite3 import SQLITE_ALTER_TABLE
from turtle import window_height
import numpy as np
import pygame
from scipy.signal import convolve2d
import sys
import time

class Connect4():
    def __init__(self):
        self.gameOver = False
        self.isDraw = False
        self.turn = -1
        self.boardRows = 6
        self.boardCols = 7
        self.winCount = 4
        self.squareSize = 100
        self.pieceRadius = int(self.squareSize/2 - 5)
        self.windowWidth = (self.boardCols + 1)* self.squareSize 
        self.windowHeight = (self.boardRows + 2) * self.squareSize
        self.winKernels = [np.array([[1, 1, 1, 1]]), \
                        np.transpose(np.array([[1, 1, 1, 1]])), \
                        np.eye(4, dtype=np.uint8), \
                        np.fliplr(np.eye(4, dtype=np.uint8))]

        self.colors = {"blue": (14, 14, 160),\
                        "red": (210, 4, 4), \
                        "green": (0, 255, 0), \
                        "yellow": (224, 217, 2), \
                        "black": (0, 0, 0), \
                        "white": (255, 255, 255), \
                        "grey100": (100, 100, 100), \
                        "grey30": (23, 23, 23), \
                        "cyan": (0,255,255)}

        self.board = np.zeros((self.boardRows, self.boardCols), dtype=np.int8)
        pygame.init()
        self.screen = pygame.display.set_mode((self.windowWidth, self.windowHeight))
        self.screen.fill(self.colors["grey30"])
        pygame.display.set_caption("Connect4")
        self.drawBoard()
        pygame.display.update()

    def drawBoard(self):
        for col in range(self.boardCols):
            for row in range(self.boardRows):
                pygame.draw.rect(self.screen, self.colors["blue"], (col * self.squareSize + (self.squareSize/2), (row + 1) * self.squareSize + (self.squareSize/2), self.squareSize, self.squareSize))
                pygame.draw.circle(self.screen, self.colors["black"], (col * self.squareSize + 2*(self.squareSize / 2), (row + 1) * self.squareSize + 2*(self.squareSize / 2)), self.pieceRadius)
        pygame.draw.rect(self.screen, self.colors["blue"], (self.boardCols*self.squareSize + (int(self.squareSize/2.5)), ((self.boardRows + 1) * self.squareSize + (self.squareSize/2)), int(self.squareSize/3), int(self.squareSize/2)))
        pygame.draw.rect(self.screen, self.colors["blue"], ((self.squareSize/2) - (int(self.squareSize/4)), ((self.boardRows + 1) * self.squareSize + (self.squareSize/2)), int(self.squareSize/3), int(self.squareSize/2)))

    def dropPiece(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.MOUSEMOTION:
                self.__handleOnMouseMotionEvent(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.__handleOnMouseClickEvent(event)
            

    def __handleOnMouseMotionEvent(self, event):
        pygame.draw.rect(self.screen, self.colors["grey30"], (0, 0, self.windowWidth + self.squareSize/2, self.squareSize + self.squareSize/2))
        if self.turn == -1:
            pygame.draw.circle(self.screen, self.colors["red"], (event.pos[0], int(self.squareSize/1.2)), self.pieceRadius)
        else:
            pygame.draw.circle(self.screen, self.colors["yellow"], (event.pos[0], int(self.squareSize/1.2)), self.pieceRadius)
        self.__drawCurrentPlayerString()

    def __handleOnMouseClickEvent(self, event):
        pygame.draw.rect(self.screen, self.colors["grey30"], (0, 0, self.windowWidth + self.squareSize/2, self.squareSize + self.squareSize/2))
        self.__playStep(event)
        if self.turn == -1:
            pygame.draw.circle(self.screen, self.colors["red"], (event.pos[0], int(self.squareSize/1.2)), self.pieceRadius)
        else:
            pygame.draw.circle(self.screen, self.colors["yellow"], (event.pos[0], int(self.squareSize/1.2)), self.pieceRadius)
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
            self.__checkForGameOver()
            if self.gameOver == True:
                self.__handleGameOver()
            self.__handleTurn()
        else:
            if self.__isDraw():
                self.__checkForGameOver()
            print("INVALID MOVE, CHOOSE AGAIN!")
        pygame.display.update()

    def __handleSelection(self, event):
        self.selection = int(event.pos[0] / (self.squareSize) + 0.5) - 1
        self.selection = self.__clampSelection(self.selection, 0, self.boardCols-1)
        if not self.__isValidSelection():
            self.selection = -1

    def __clampSelection(self, val, minVal, maxVal):
        return max(min(maxVal, val), minVal)
    
    def __isValidSelection(self):
        if (self.selection < 0 or self.selection > self.boardCols) or (self.board[self.boardRows-1][self.selection]) != 0:
            return False
        return True

    def __updateBoard(self):
        for i in range(self.boardRows):
            if self.board[i][self.selection] != 0:
                continue
            self.board[i][self.selection]  = self.turn
            if self.turn == -1:
                color = self.colors["red"]
            else:
                color = self.colors["yellow"]
            pygame.draw.circle(self.screen, color, (self.selection * self.squareSize + (self.squareSize), (self.boardRows - i) * self.squareSize + (self.squareSize)), self.pieceRadius)
            break

    def __printBoard(self):
        print(np.flip(self.board, 0))

    def __checkForGameOver(self):
        for kernel in self.winKernels:
            if (convolve2d(self.board == self.turn, kernel, mode = "valid") == 4).any():
                self.gameOver = True

        if self.__isDraw():
            self.isDraw = True
            self.gameOver = True

    def __handleTurn(self):
        if self.turn == -1:
            self.turn = 1
        else:
            self.turn = -1

    def __handleGameOver(self):
        if self.isDraw:
            self.__handleDraw()
        else:
            self.__handleWin()
        pygame.display.update()


    def __isDraw(self):
        if self.validActions() == [] and self.gameOver == False:
            return True
        return False

    def __handleDraw(self):
        print("***GAME OVER***")
        print("IT'S A DRAW!!")
        myFont = pygame.font.SysFont("monospace", 50, bold=True)
        label = myFont.render("IT'S A DRAW!!", 1, self.colors["green"])
        pygame.draw.rect(self.screen, self.colors["grey30"], (160, 250, 400, 50))
        self.screen.blit(label, (165, 250))

    def __handleWin(self):
        winner, color = self.__handleWinnerString()
        print("***GAME OVER***")
        print("WINNER: ", winner)
        myFont = pygame.font.SysFont("monospace", 50, bold=True)
        label = myFont.render("WINNER: " + winner + "!!", 1, color)
        pygame.draw.rect(self.screen, self.colors["grey30"], (130, 250, 540, 55))
        self.screen.blit(label, (135, 250))

    def __handleWinnerString(self):
        if self.turn == -1:
            playerString = "PLAYER 1"
            color = self.colors["red"]
        else:
            playerString ="PLAYER 2"
            color = self.colors["yellow"]
        return playerString, color
    
    def __handleCurrentPlayerString(self):
        if self.turn == -1:
            playerString = "PLAYER 1"
            color = self.colors["red"]
        else:
            playerString = "PLAYER 2"
            color = self.colors["yellow"]
        return playerString, color

    def validActions(self):
        actions = []
        for col in range(self.boardCols):
            if self.board[self.boardRows-1][col] == 0:
                actions.append(col)
        return actions

    def updateBoard(self, move):
        ###Assume the move selected is valid
        for i in range(self.boardRows):
            if self.board[i][move] != 0:
                continue
            self.board[i][move]  = self.turn
            if self.turn == -1:
                color = self.colors["red"]
            else:
                color = self.colors["yellow"]
            pygame.draw.circle(self.screen, color, (move * self.squareSize + (self.squareSize / 2), (self.boardRows - i) * self.squareSize + (self.squareSize / 2)), self.pieceRadius)
            break

if __name__ == "__main__":
    while (True):
        gameInstance = Connect4()
        while not gameInstance.gameOver:
            gameInstance.dropPiece()
            pygame.display.update()
        pygame.time.wait(3000)
        pygame.quit()

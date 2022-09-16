
from sqlite3 import SQLITE_ALTER_TABLE
import numpy as np
import pygame
from scipy.signal import convolve2d
import sys
import time

CLR_BLUE = (0, 0, 255)
CLR_RED = (255, 0, 0)
CLR_YELLOW = (255, 255, 0)
CLR_BLACK = (0, 0, 0)
CLR_GREEN = (0, 255, 0)

BOARD_ROWS = 6
BOARD_COLS = 7
WIN_COUNT = 4

SQUARE_SIZE = 100
PIECE_RADIUS = int(SQUARE_SIZE/2 - 5)
WINDOW_WIDTH = BOARD_COLS * SQUARE_SIZE
WINDOW_HEIGHT = (BOARD_ROWS + 1) * SQUARE_SIZE

PLAYER_1_TURN = -1
PLAYER_2_TURN = 1

gameOver = False
board = []
turn = PLAYER_1_TURN

horizontalWinKernel = np.array([[1, 1, 1, 1]])
verticalWinKernel = np.transpose(horizontalWinKernel)

diagonalWinKernelOne = np.eye(4, dtype=np.uint8)
diagonalWinkernelTwo = np.fliplr(diagonalWinKernelOne)

winDetectionKernels = [horizontalWinKernel, verticalWinKernel, diagonalWinKernelOne, diagonalWinkernelTwo]

def main():
    global gameOver
    global turn
    while(True):
        gameOver = False
        draw = False
        waitForInput = False
        turn = -1
        screen = setup()
        while not gameOver:
            waitForInput = True
            handlePygameEvents(screen, waitForInput)
            pygame.display.update()
            if(isDraw()):
                draw = True
                gameOver=True
        waitForInput = False
        if draw:
            handleDraw(screen)
        else:
            handleWin(screen)
        pygame.display.update()
        pygame.time.wait(3000)
        pygame.quit()

def handleWin(screen):
    winner, color = handlePlayerString()
    print("***GAME OVER***")
    print("WINNER: ", winner)
    myFont = pygame.font.SysFont("monospace", 50, bold=True)
    label = myFont.render("WINNER: " + winner + "!!", 1, color)
    pygame.draw.rect(screen, CLR_BLACK, (95, 250, 540, 50))
    screen.blit(label, (100, 250))

def handleDraw(screen):
    print("***GAME OVER***")
    print("IT'S A DRAW!!")
    myFont = pygame.font.SysFont("monospace", 50, bold=True)
    label = myFont.render("IT'S A DRAW!!", 1, CLR_GREEN)
    pygame.draw.rect(screen, CLR_BLACK, (160, 250, 400, 50))
    screen.blit(label, (165, 250))

def isDraw():
    global gameOver
    if np.all(board != 0) and gameOver == False:
        return True
    return False

def playStep(event, screen):
    global turn
    global gameOver
    selection = handleSelection(event)
    if selection != -1:
        updateBoard(selection, screen)
        printBoard()
        gameOver = checkForWin()
        if(gameOver):
            return
        handleTurn()
    else:
        print("INVALID MOVE, CHOOSE AGAIN")

def handlePygameEvents(screen, waitForInput):
    if waitForInput:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(screen, CLR_BLACK, (0, 0, WINDOW_WIDTH, SQUARE_SIZE))
                if turn == PLAYER_1_TURN:
                    pygame.draw.circle(screen, CLR_RED, (event.pos[0], int(SQUARE_SIZE/2)), PIECE_RADIUS)
                else:
                    pygame.draw.circle(screen, CLR_YELLOW, (event.pos[0], int(SQUARE_SIZE/2)), PIECE_RADIUS)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                playStep(event, screen)
                if turn == PLAYER_1_TURN:
                    pygame.draw.circle(screen, CLR_RED, (event.pos[0], int(SQUARE_SIZE/2)), PIECE_RADIUS)
                else:
                    pygame.draw.circle(screen, CLR_YELLOW, (event.pos[0], int(SQUARE_SIZE/2)), PIECE_RADIUS)

def checkForWin():
    global board
    for kernel in winDetectionKernels:
        if (convolve2d(board == turn, kernel, mode = "valid") == WIN_COUNT).any():
            return True
    return False
    
def printBoard():
    global board
    print(np.flip(board, 0))


def isValidSelection(selection):
    global board
    try:
        int(selection)
    except ValueError:
        return False
    selection = int(selection)
    if selection < 0 or selection > BOARD_COLS or board[BOARD_ROWS-1][selection] != 0:
        return False

    return True


def handlePlayerString():
    global turn
    if turn == PLAYER_1_TURN:
        playerString = "PLAYER 1"
        color = CLR_RED
    else:
        playerString = "PLAYER 2"
        color = CLR_YELLOW
    return playerString, color

def handleSelection(event):

    player = handlePlayerString()
    selection = int(event.pos[0] / SQUARE_SIZE)
    if not isValidSelection(selection):
        return -1
    else:
        return int(selection)

def handleTurn():
    global turn
    if turn == PLAYER_1_TURN:
        turn = PLAYER_2_TURN
    else:
        turn = PLAYER_1_TURN
    
def updateBoard(selection,screen):
    for i in range(BOARD_ROWS):
        if board[i][selection] != 0:
            continue
        board[i][selection] = turn
        if turn == PLAYER_1_TURN:
            color = CLR_RED
        else:
            color = CLR_YELLOW
        pygame.draw.circle(screen, color, (selection * SQUARE_SIZE + SQUARE_SIZE / 2, (BOARD_ROWS - i) * SQUARE_SIZE + SQUARE_SIZE/2), PIECE_RADIUS)
        break

def createBoard(rows, cols):
    global board
    board = np.zeros((rows,cols),dtype=np.int8)

def setup():
    createBoard(BOARD_ROWS, BOARD_COLS)
    pygame.init()    
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Connect4")
    drawBoard(screen)
    return screen

def drawBoard(screen):
    for col in range (BOARD_COLS):
        for row in range(BOARD_ROWS):
            pygame.draw.rect(screen, CLR_BLUE, (col * SQUARE_SIZE, (row + 1) * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
            pygame.draw.circle(screen, CLR_BLACK, (col * SQUARE_SIZE + SQUARE_SIZE / 2, (row + 1) * SQUARE_SIZE + SQUARE_SIZE/2), PIECE_RADIUS)
    pygame.display.update()




if __name__ == "__main__":
    main()
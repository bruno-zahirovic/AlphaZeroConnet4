import numpy as np
import torch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

from tictactoe import TicTacToe


class ResNet(nn.Module):
    def __init__(self, game, numResBlocks, numHiddenLayers, device):
        super().__init__()
        self.device = device

        self.startBlock = nn.Sequential(
            nn.Conv2d(3, numHiddenLayers, kernel_size=3, padding=1),
            nn.BatchNorm2d(numHiddenLayers),
            nn.ReLU()
        )

        self.backBone = nn.ModuleList(
            [ResBlock(numHidden=numHiddenLayers) for i in range(numResBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(numHiddenLayers, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.rowCount * game.colCount, game.actionSize)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(numHiddenLayers, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.rowCount * game.colCount, 1),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)

        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value


class ResBlock(nn.Module):
    def __init__(self, numHidden):
        super().__init__()
        self.conv1 = nn.Conv2d(numHidden, numHidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(numHidden)
        self.conv2 = nn.Conv2d(numHidden, numHidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(numHidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x
        

def main():
    tictactoe = TicTacToe()

    state = tictactoe.GetInitialState()
    state = tictactoe.GetNextState(state, 2, -1)
    state = tictactoe.GetNextState(state, 4, -1)
    state = tictactoe.GetNextState(state, 6, 1)
    state = tictactoe.GetNextState(state, 8, 1)


    encoded_state = tictactoe.GetEncodedState(state)

    tensor_state = torch.tensor(encoded_state).unsqueeze(0)

    model = ResNet(tictactoe, 4, 64, device=torch.device("cpu"))
    model.load_state_dict(torch.load('model_2.pt'))
    model.eval()

    policy, value = model(tensor_state)
    value = value.item()
    policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

    print(value)

    print(state)
    print(tensor_state)

    plt.bar(range(tictactoe.actionSize), policy)
    plt.show()


if __name__ == "__main__":
    main()
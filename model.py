from platform import architecture
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Connect4Model(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        ##define layers

        #conv
        self.convLayer = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, 
                                    stride = 1, bias=True, padding=1 , padding_mode='zeros')
        self.batchNormLayer = nn.BatchNorm2d(128)
        #Res block 1
        self.resBlockOneConvOne = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, \
                                    stride = 1, bias=False, padding=1 , padding_mode='zeros')
        self.resBlockOneBNOne = nn.BatchNorm2d(128)
        self.resBlockOneConvTwo = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, \
                                    stride = 1, bias=False, padding=1 , padding_mode='zeros')
        self.resBlockOneBNTwo = nn.BatchNorm2d(128)
        #Res block 2
        self.resBlockTwoConvOne = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, \
                                    stride = 1, bias=False, padding=1 , padding_mode='zeros')
        self.resBlockTwoBNOne = nn.BatchNorm2d(128)
        self.resBlockTwoConvTwo = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, \
                                    stride = 1, bias=False, padding=1 , padding_mode='zeros')
        self.resBlockTwoBNTwo = nn.BatchNorm2d(128)
        #Res block 3
        self.resBlockThreeConvOne = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, \
                                    stride = 1, bias=False, padding=1 , padding_mode='zeros')
        self.resBlockThreeBNOne = nn.BatchNorm2d(128)
        self.resBlockThreeConvTwo = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, \
                                    stride = 1, bias=False, padding=1 , padding_mode='zeros')
        self.resBlockThreeBNTwo = nn.BatchNorm2d(128)
        #Res block 4
        self.resBlockFourConvOne = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, \
                                    stride = 1, bias=False, padding=1 , padding_mode='zeros')
        self.resBlockFourBNOne = nn.BatchNorm2d(128)
        self.resBlockFourConvTwo = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, \
                                    stride = 1, bias=False, padding=1 , padding_mode='zeros')
        self.resBlockFourBNTwo = nn.BatchNorm2d(128)
        #Res block 5
        self.resBlockFiveConvOne = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, \
                                    stride = 1, bias=False, padding=1 , padding_mode='zeros')
        self.resBlockFiveBNOne = nn.BatchNorm2d(128)
        self.resBlockFiveConvTwo = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, \
                                    stride = 1, bias=False, padding=1 , padding_mode='zeros')
        self.resBlockFiveBNTwo = nn.BatchNorm2d(128)
        #value head
        self.valueConv = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=1, \
                                    stride = 1, bias=True)
        self.valueBN = nn.BatchNorm2d(3)
        self.valueFullyConnected = nn.Linear(in_features=3*6*7, out_features=32)
        self.valueHead = nn.Linear(in_features=32,out_features=1)
        #policy head
        self.policyConv = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, \
                                    stride = 1, bias=True)
        self.policyBN = nn.BatchNorm2d(32)
        self.policyHead = nn.Linear(in_features=32*6*7, out_features=7)
        self.policyLSM = nn.LogSoftmax(dim=1)

        self.to(device)

    def forward(self, x):
        #define connections between the layers
        # x shape = (3, 6, 7)

        #add dimensions for batch size
        x = x.view(-1, 3, 6, 7)
        x = F.relu(self.batchNormLayer(self.convLayer(x)))

        #Res block 1
        res = x
        x = F.relu(self.resBlockOneBNOne(self.resBlockOneConvOne(x)))
        x = F.relu(self.resBlockOneBNTwo(self.resBlockOneConvTwo(x)))
        x += res
        x = F.relu(x)

        #Res block 2
        res = x
        x = F.relu(self.resBlockThreeBNOne(self.resBlockThreeConvOne(x)))
        x = F.relu(self.resBlockThreeBNTwo(self.resBlockThreeConvTwo(x)))
        x += res
        x = F.relu(x)

        #Res block 3
        res = x
        x = F.relu(self.resBlockFourBNOne(self.resBlockFourConvOne(x)))
        x = F.relu(self.resBlockFourBNTwo(self.resBlockFourConvTwo(x)))
        x += res
        x = F.relu(x)

        #Res block 4
        res = x
        x = F.relu(self.resBlockFiveBNOne(self.resBlockFiveConvOne(x)))
        x = F.relu(self.resBlockFiveBNTwo(self.resBlockFiveConvTwo(x)))
        x += res
        x = F.relu(x)

        #Res block 5
        res = x
        x = F.relu(self.resBlockTwoBNOne(self.resBlockTwoConvOne(x)))
        x = F.relu(self.resBlockTwoBNTwo(self.resBlockTwoConvTwo(x)))
        x += res
        x = F.relu(x)

        #Value head
        value = F.relu(self.valueBN(self.valueConv(x)))
        value = value.view(-1, 3*6*7)
        value = F.relu(self.valueFullyConnected(value))
        value = torch.tanh(value)

        #Policy head
        policy = F.relu(self.policyBN(self.policyConv(x)))
        policy = policy.view(-1, 32*6*7)
        policy = F.relu(self.policyHead(policy))
        policy = self.policyLSM(policy).exp()

        return value, policy

if __name__ == "__main__":
    from torchinfo import summary
    if torch.cuda.is_available():
        print(torch.zeros(1).cuda())
        print(torch.cuda.get_device_name(0))
        device = torch.device('cuda')

    else:
        torch.zeros(1).cuda()
        exit()

    model = Connect4Model(device=device)
    archSummary = summary(model, input_size=(16,3,6,7), verbose=0)

    print(archSummary)
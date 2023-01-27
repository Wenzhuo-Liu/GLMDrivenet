import torch
import math
from torch import nn
from FC_attention import *
# from EcaBlock import *

class MSFRF(nn.Module):
    
    def __init__(self, channels=128,r=4):
        super(MSFRF, self).__init__()
        inter_channels = int(channels // r) 
        self.conv1 = nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.local_att1 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.conv = nn.Conv1d(1, 1, kernel_size = 3, padding = (3 - 1) // 2, bias = False)
        self.bn = nn.BatchNorm2d(channels)
 
        self.attention2 = FC_Block(channels, 16)
 
        self.sigmoid = nn.Sigmoid()
 
 
    def forward(self, x, residual):
        xa = x + residual

        xz1 = self.local_att1(xa)

        xz2 = self.conv(xa.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        xz2 = self.bn(xz2)

        xlg1 = xz2 + xz1
        xlg1 = self.attention2(xlg1)
        wei = self.sigmoid(xlg1)
 
 
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

if __name__ == "__main__":

    model = MSFRF()#.cuda()
    print("Model loaded.")
    image = torch.rand(2, 128,1,1)#.cuda()
    audio = torch.rand(2, 128,1,1)#.cuda()
    print("Image and audio loaded.")

	# Run a feedforward and check shape
    c = model(image,audio)
    print(image.shape)
    print(c.shape)
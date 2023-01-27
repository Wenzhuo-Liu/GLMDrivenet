import torch
import torch.nn as nn


class FC_Block(nn.Module):
    def __init__(self, channels, ratio):
        super(FC_Block, self).__init__()
        # self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        # self.max_pooling = nn.AdaptiveMaxPool2d(1)
        # if mode == "max":
        #     self.global_pooling = self.max_pooling
        # elif mode == "avg":
        #     self.global_pooling = self.avg_pooling
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features = channels, out_features = channels // ratio, bias = False),
            nn.ReLU(),
            nn.Linear(in_features = channels // ratio, out_features = channels, bias = False),
        )
        self.sigmoid = nn.Sigmoid()
     
    
    def forward(self, x):
        b, c, _, _ = x.shape
        # v = self.global_pooling(x).view(b, c)
        v = x.view(b,c)
        v = self.fc_layers(v).view(b, c, 1, 1)
        v = self.sigmoid(v)
        return x*v

if __name__ == "__main__":
    model = FC_Block(3, 16)
    feature_maps = torch.randn((2, 3, 1, 1))
    model(feature_maps)

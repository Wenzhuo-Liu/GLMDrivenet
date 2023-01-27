import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class GLIBlock(nn.Module):
    def __init__(self, channels, ratio,gamma = 2, b = 1):
        super(GLIBlock, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features = channels, out_features = channels // ratio, bias = False),
            nn.ReLU(),
            nn.Linear(in_features = channels // ratio, out_features = channels, bias = False)
        )
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.conv = nn.Conv1d(1, 1, kernel_size = kernel_size, padding = (kernel_size - 1) // 2, bias = False)

        self.cfc = Parameter(torch.Tensor(channels, 2))
        self.cfc.data.fill_(0)

        self.bn = nn.BatchNorm2d(channels)
        self.activation = nn.Sigmoid()

        setattr(self.cfc, 'srm_param', True)
        setattr(self.bn.weight, 'srm_param', True)
        setattr(self.bn.bias, 'srm_param', True)

        self.sigmoid = nn.Sigmoid()

    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 2 对应元素相乘
        z = torch.sum(z, dim=2)[:, :, None, None] # B x C x 1 x 1

        z_hat = self.bn(z)
        g = self.sigmoid(z_hat)

        return g

    def forward(self, x, eps=1e-5):
        b, c, h, w = x.shape # x.shape = torch.Size([8, 512, 14, 14])

        # 全局最大池化
        avg_x_fc = self.avg_pooling(x).view(b, c) # [8, 512]

        # 全局最大池化 + 全局平均池化 + 全局标准差池化(conv1d)
        avg_x_conv = self.avg_pooling(x) # [8, 512,1,1]

        # 特征提取层
        v_fc = self.fc_layers(avg_x_fc)# + self.fc_layers(max_x_fc)  + self.fc_layers(std_x_fc) # [8,512]
        v_conv = self.conv(avg_x_conv.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)# + self.conv(max_x_conv.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) + self.conv(std_x_conv.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # [8,512,1,1]

        v_x,v_y = v_fc.shape # 8,512
        v_fc = v_fc.view(v_x,v_y,1) # [8, 512, 1]
        v_conv = v_conv.view(v_x,v_y,1) # [8, 512, 1]

        # 通道平均池化
        v_sum = torch.cat((v_fc, v_conv), dim=2)
        v_sum = self._style_integration(v_sum)
        # print(v_sum.size()) # [2, 3, 2]
        # v_sum = (v_fc + v_conv)/2

        # v = self.sigmoid(v_sum) # [8, 512, 1, 1]
         # [8, 512, 14, 14]
        return x * v_sum

if __name__ == "__main__":
    model = GLIBlock(512, 16,gamma = 2, b = 1)
    feature_maps1 = torch.randn((2, 3, 2, 2))
    feature_maps = torch.randn((8, 512, 14, 14))
    a= model(feature_maps)
import torch, os
from torch.optim import *
from torch.autograd import *
from torch import nn
from torch.nn import functional as F
from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from GLI_CAM import GLIBlock

class ImageConvNet(nn.Module):

	def __init__(self):
		super(ImageConvNet, self).__init__()
		self.pool = nn.MaxPool2d(2, stride=2)
		
		self.cnn1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)
		self.cnn2 = nn.Conv2d(64, 64, 3, padding=1)
		self.bat10 = nn.BatchNorm2d(64)
		self.bat11 = nn.BatchNorm2d(64)

		self.cnn3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
		self.cnn4 = nn.Conv2d(128, 128, 3, padding=1)
		self.bat20 = nn.BatchNorm2d(128)
		self.bat21 = nn.BatchNorm2d(128)

		self.cnn5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
		self.cnn6 = nn.Conv2d(256, 256, 3, padding=1)
		self.bat30 = nn.BatchNorm2d(256)
		self.bat31 = nn.BatchNorm2d(256)

		self.cnn7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
		self.cnn8 = nn.Conv2d(512, 512, 3, padding=1)
		self.bat40 = nn.BatchNorm2d(512)
		self.bat41 = nn.BatchNorm2d(512)

		self.SeBlock1 = GLIBlock(64, 16, gamma = 2, b = 1)
		self.SeBlock2 = GLIBlock(64, 16, gamma = 2, b = 1)
		self.SeBlock3 = GLIBlock(128, 16, gamma = 2, b = 1)
		self.SeBlock4 = GLIBlock(128, 16, gamma = 2, b = 1)
		self.SeBlock5 = GLIBlock(256, 16, gamma = 2, b = 1)
		self.SeBlock6 = GLIBlock(256, 16, gamma = 2, b = 1)
		self.SeBlock7 = GLIBlock(512, 16, gamma = 2, b = 1)
		self.SeBlock8 = GLIBlock(512, 16, gamma = 2, b = 1)
		
	def forward(self, inp):
		c = F.relu(self.SeBlock1(self.bat10(self.cnn1(inp))))
		c = F.relu(self.SeBlock2(self.bat11(self.cnn2(c))))
		c = self.pool(c)

		c = F.relu(self.SeBlock3(self.bat20(self.cnn3(c))))
		c = F.relu(self.SeBlock4(self.bat21(self.cnn4(c))))
		c = self.pool(c)
		
		c = F.relu(self.SeBlock5(self.bat30(self.cnn5(c))))
		c = F.relu(self.SeBlock6(self.bat31(self.cnn6(c))))
		c = self.pool(c)
		
		c = F.relu(self.SeBlock7(self.bat40(self.cnn7(c))))
		c = F.relu(self.SeBlock8(self.bat41(self.cnn8(c))))
		return c

	# Dummy function, just to check if feedforward is working or not
	def loss(self, output):
		return (output.mean())**2


if __name__ == "__main__":
	model = ImageConvNet().cuda()
	print("Model loaded.")
	image = Variable(torch.rand(2, 3, 224, 224)).cuda()
	print("Image loaded.")

	# Run a feedforward and check shape
	c = model(image)
	print(image.shape)
	print(c.shape)

from image_convnet import *
from audio_convnet import *
from torch.nn.parameter import Parameter
from GLI_CAM import GLIBlock
from MS_FRF import *

from utils.mydata_xu import *
import torch.nn.functional as F
## Main NN starts here
class AVENet1(nn.Module):

	def __init__(self):
		super(AVENet1, self).__init__()

		self.relu   = F.relu
		self.imgnet = ImageConvNet()
		self.audnet = AudioConvNet()
		self.fusion = MSFRF()

		# Vision subnetwork
		self.vpool4  = nn.MaxPool2d(14, stride=14)
		self.vfc1    = nn.Linear(512, 128)
		self.vfc2    = nn.Linear(128, 128)
		self.vl2norm = nn.BatchNorm1d(128)

		# Audio subnetwork
		self.apool4  = nn.MaxPool2d((16, 12), stride=(16, 12))
		self.afc1    = nn.Linear(512, 128)
		self.afc2    = nn.Linear(128, 128)
		self.al2norm = nn.BatchNorm1d(128)

		# Combining layers
		self.mse     = F.mse_loss
		#self.fc3     = nn.Linear(1, 2)
		self.fc3     = nn.Linear(128, 3)
		self.softmax = F.softmax


	def forward(self, image, audio):
		# Image
		img = self.imgnet(image)
		img = self.vpool4(img).squeeze(2).squeeze(2)
		img = self.relu(self.vfc1(img))
		img = self.vfc2(img)
		n,c = img.shape
		img = self.vl2norm(img).view(n,c,1,1)

		# Audio
		aud = self.audnet(audio)
		aud = self.apool4(aud).squeeze(2).squeeze(2)
		aud = self.relu(self.afc1(aud))
		aud = self.afc2(aud)
		aud = self.al2norm(aud).view(n,c,1,1)

		out = self.fusion(img,aud)
		out = out.squeeze(2).squeeze(2)
		out = self.fc3(out)
		
		return out, img, aud

        # # join
		#  # 获取批次大小

        # # 扩充一维
		# img = img.cuda()
		# img = torch.cat([img,torch.ones(n,1).cuda()],dim=1) # (16*129)
		# aud = torch.cat([aud,torch.ones(n,1).cuda()],dim=1)

        # # 并行提取各模态特征
		# fusion_img = torch.matmul(img,self.w_img) # (4,16,8)
		# fusion_aud = torch.matmul(aud,self.w_aud)

        # # 最终融合
		# fusion_img_aud = fusion_img * fusion_aud # (4,16,8) 对应位置相乘
		# fusion_img_aud = torch.matmul(self.w_f,fusion_img_aud.permute(1,0,2)).squeeze() + self.bias

        # # 输出结果
		# out = self.fc3(fusion_img_aud)

		# return out, img, aud


	def get_image_embeddings(self, image):
		# Just get the image embeddings
		img = self.imgnet(image)
		img = self.vpool4(img).squeeze(2).squeeze(2)
		img = self.relu(self.vfc1(img))
		img = self.vfc2(img)
		img = self.vl2norm(img)
		return img

def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)

if __name__ == '__main__':
	# Bias = torch.tensor([[1.8062e-25, 7.3008e-43, 1.8062e-25, 7.3008e-43, 3.2415e-24, 7.3008e-43,
    #      1.8062e-25, 7.3008e-43]], requires_grad=True).cuda()
	# f = torch.tensor([[0., 0., 0., 0.]], requires_grad=True).cuda()
	model = AVENet1().cuda()
	param_size, param_sum, buffer_size, buffer_sum, all_size = getModelSize(model)
	print(param_size,param_sum, buffer_size, buffer_sum, all_size)

	
	image = Variable(torch.rand(2, 3, 224, 224)).cuda()
	speed = Variable(torch.rand(2, 1, 257, 200)).cuda()

	# Run a feedforward and check shape
	o,_,_ = model(image,speed)
	# W_img = model.w_img
	# W_aud = model.w_aud
	# W_f = model.w_f
	# Bias = model.bias
	# print(W_img)
	# print(W_aud)
	# print(W_f)
	# print(Bias)
	# print(o)
	# o=F.softmax(o,1)
	# print(o)
	print(o.shape)#[2,3]
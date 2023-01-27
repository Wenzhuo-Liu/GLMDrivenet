from image_convnet import *
from audio_convnet import *
from AVENet_yuan import *
import shutil
import time
import argparse
from torch.optim import *
from torchvision.transforms import *
import warnings
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import json
from utils.mydata_xu import *
import math
from tqdm import tqdm
from prettytable import PrettyTable

choices = ["demo", "main", "test","checkValidation", "getVideoEmbeddings", "generateEmbeddingsForVideoAudio", \
			"imageToImageQueries", "crossModalQueries"]

parser = argparse.ArgumentParser(description="Select code to run.")
parser.add_argument('--mode', default="test", choices=choices, type=str)

checkpoint_dir='/root/GLMDrivenet/model_save/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class valConfusionMatrix(object):
	def __init__(self, num_classes: int, labels: list):
		self.matrix = np.zeros((num_classes, num_classes))
		self.num_classes = num_classes
		self.labels = labels
	def update(self, preds, labels):
		for p, t in zip(preds, labels):
			self.matrix[p, t] += 1
	def summary(self):
		f1_list=[]
		for i in range(self.num_classes):
			TP = self.matrix[i, i]
			FP = np.sum(self.matrix[i, :]) - TP
			FN = np.sum(self.matrix[:, i]) - TP
			TN = np.sum(self.matrix) - TP - FP - FN
			Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
			Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
			#Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
			F1=round(2*Precision*Recall / (Precision+Recall), 3) if Precision+Recall != 0 else 0.
			f1_list.append(F1)
		return f1_list

class testConfusionMatrix(object):
	def __init__(self, num_classes: int, labels: list):
		self.matrix = np.zeros((num_classes, num_classes))
		self.num_classes = num_classes
		self.labels = labels

	def update(self, preds, labels):
		for p, t in zip(preds, labels):
			self.matrix[p, t] += 1

	def summary(self):
		# calculate accuracy
		sum_TP = 0
		for i in range(self.num_classes):
			sum_TP += self.matrix[i, i]
		acc = sum_TP / np.sum(self.matrix)
		print("the model accuracy is ", acc)

		# precision, recall, specificity
		table = PrettyTable()
		table.field_names = ["", "Precision", "Recall", "Specificity","F1"]
		for i in range(self.num_classes):
			TP = self.matrix[i, i]
			FP = np.sum(self.matrix[i, :]) - TP
			FN = np.sum(self.matrix[:, i]) - TP
			TN = np.sum(self.matrix) - TP - FP - FN
			Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
			Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
			Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
			F1=round(2*Precision*Recall / (Precision+Recall), 3) if Precision+Recall != 0 else 0.
			table.add_row([self.labels[i], Precision, Recall, Specificity,F1])
		print(table)

	def plot(self):
		matrix = self.matrix
		print(matrix)
		plt.imshow(matrix, cmap=plt.cm.Blues)

		# 设置x轴坐标label
		plt.xticks(range(self.num_classes), self.labels, rotation=45)
		# 设置y轴坐标label
		plt.yticks(range(self.num_classes), self.labels)
		# 显示colorbar
		plt.colorbar()
		plt.xlabel('True Labels')
		plt.ylabel('Predicted Labels')
		plt.title('Confusion matrix')

		# 在图中标注数量/概率信息
		thresh = matrix.max() / 2
		for x in range(self.num_classes):
			for y in range(self.num_classes):
				# 注意这里的matrix[y, x]不是matrix[x, y]
				info = int(matrix[y, x])
				plt.text(x, y, info,
						verticalalignment='center',
						horizontalalignment='center',
						color="white" if info > thresh else "black")
		plt.tight_layout()
		plt.show()

# Demo to check if things are working
def demo():
	model = AVENet1()
	image = Variable(torch.rand(2, 3, 224, 224))
	audio = Variable(torch.rand(2, 1, 257, 200))

	out, v, a = model(image, audio)
	print(image.shape, audio.shape)
	print(v.shape, a.shape, out.shape)

class LossAverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

class AccAverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val
		self.count += n
	def getacc(self):
		return (self.sum *100) / self.count

# Main function here
def main(use_cuda=True, EPOCHS=200, batch_size=8, model_name="avenet.pt"):
	
	model = getAVENet(use_cuda)

	# Load from before
	if os.path.exists(model_name):
		model.load_state_dict(torch.load(model_name))
		print("Loading from previous checkpoint.")

	# list_image1=getimage()
	dataset = Mydata(img_speed_path='/root/autodl-tmp/motor/train.txt',
    img_path='/root/autodl-tmp/motor_dataset/train/img/',
    speed_path='/root/autodl-tmp/motor_dataset/train/speed/')

	valdataset = Mydata(img_speed_path='/root/autodl-tmp/motor/val.txt',
    img_path='/root/autodl-tmp/motor_dataset/val/img/',
    speed_path='/root/autodl-tmp/motor_dataset/val/speed/')
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
	valdataloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True, num_workers=2)

	crossEntropy = nn.CrossEntropyLoss()
	print("Loaded dataloader and loss function.")

	# optim = Adam(model.parameters(), lr=lr, weight_decay=1e-7)
	optim = SGD(model.parameters(), lr=0.25e-3, momentum=0.9, weight_decay=1e-4)
	print("Optimizer loaded.")
	model.train()

	try:
		best_precision = 0
		lowest_loss = 100000
		best_avgf1=0
		best_weightf1=0
		for epoch in range(EPOCHS):
			if(50<=epoch<100):
				optim = SGD(model.parameters(), lr=0.25e-4, momentum=0.9, weight_decay=1e-4)
			if(epoch>=100):
				optim = SGD(model.parameters(), lr=0.25e-5, momentum=0.9, weight_decay=1e-4)
			# Run algo

			train_losses = LossAverageMeter()
			train_acc = AccAverageMeter()
			if (epoch == 0):
				end = time.time()
			for subepoch, (img, aud, out) in enumerate(dataloader):
				# print(subepoch)
				# print(img.shape)
				# print(img.shape)
				# print(aud.shape)
				if(epoch==0 and subepoch==0):
					print(time.time() - end)
				optim.zero_grad()				
				# Filter the bad ones first
				out = out.squeeze(1)
				idx = (out != 3).numpy().astype(bool)				
				if idx.sum() == 0:
					continue
				# Find the new variables
				img = torch.Tensor(img.numpy()[idx, :, :, :])
				aud = torch.Tensor(aud.numpy()[idx, :, :, :])
				out = torch.LongTensor(out.numpy()[idx])
				# Print shapes
				img = Variable(img)
				aud = Variable(aud)
				out = Variable(out)
				# print(img.shape, aud.shape, out.shape)
				M = img.shape[0]
				if use_cuda:
					img = img.cuda()
					aud = aud.cuda()
					out = out.cuda()
				# img=img.to(device)
				# aud=aud.to(device)
				o, _, _ = model(img, aud)
				# if subepoch%400 == 0:
				# 	print(o)
				# 	print(out)
				# print(o.shape, out.shape)
				loss = crossEntropy(o, out)
				# print(loss)
				train_losses.update(loss.item(),M)
				loss.backward()
				optim.step()
				# Calculate accuracy
				o=F.softmax(o,1)
				_, ind = o.max(1)
				accuracy = (ind.data == out.data).sum()*1.0/M
				train_acc.update((ind.data == out.data).sum()*1.0,M)

				if subepoch%400 == 0:
					print("Epoch: %d, Subepoch: %d, Loss: %f, batch_size: %d,acc: %f, zongacc: %f" % (epoch, subepoch, train_losses.avg, M, accuracy,train_acc.getacc()))
					with open(file="/root/GLMDrivenet/log_motor/train.txt", mode="a+") as f:
						f.write("Epoch: %d, Subepoch: %d, Loss: %f, batch_size: %d, acc: %f,zongacc: %f"%(epoch, subepoch, train_losses.avg, M,accuracy, train_acc.getacc()))
			print("Epoch: %d, Loss: %f, sum: %d, acc: %f\n"%(epoch, train_losses.avg, train_losses.count, train_acc.getacc()))
			with open(file="/root/GLMDrivenet/log_motor/train.txt", mode="a+") as f:
				f.write("Epoch: %d, Loss: %f, sum: %d, acc: %f\n"%(epoch, train_losses.avg, train_losses.count, train_acc.getacc()))
			val_losses = LossAverageMeter()
			val_acc = AccAverageMeter()
			labels=['Normal','Aggressive','Drowsy']
			valconfusion = valConfusionMatrix(num_classes=3, labels=labels)
			model.eval()
			for sepoch,(img, aud, out) in enumerate(valdataloader):
				out = out.squeeze(1)
				idx = (out != 3).numpy().astype(bool)
				if idx.sum() == 0:
					continue
					# Find the new variables
				img = torch.Tensor(img.numpy()[idx, :, :, :])
				aud = torch.Tensor(aud.numpy()[idx, :, :, :])
				out = torch.LongTensor(out.numpy()[idx])
				img = Variable(img, volatile=True)
				aud = Variable(aud, volatile=True)
				out = Variable(out, volatile=True)
					# print(img.shape, aud.shape, out.shape)
				M = img.shape[0]
				if use_cuda:
					img = img.cuda()
					aud = aud.cuda()
					out = out.cuda()
				with torch.no_grad():
					o, _, _ = model(img, aud)
					valloss = crossEntropy(o, out)
				val_losses.update(valloss.item(),M)
					# Calculate valaccuracy
				o=F.softmax(o,1)
				_, ind = o.max(1)
				valconfusion.update(ind.to("cpu").numpy(), out.to("cpu").numpy())
				valaccuracy = (ind.data == out.data).sum()*1.0/M
				val_acc.update((ind.data == out.data).sum()*1.0,M)
				if sepoch%400==0:
					print("Epoch: %d, Sepoch: %d, Valloss: %f, batch_size: %d,  valacc: %f, zongvalacc: %f"%(epoch, sepoch, val_losses.avg, M, valaccuracy,val_acc.getacc()))
					with open(file="/root/GLMDrivenet/log_motor/val.txt", mode="a+") as f:
						f.write("Epoch: %d, Sepoch: %d, Valloss: %f, batch_size: %d,  valacc: %f, zongvalacc: %f"%(epoch, sepoch, val_losses.avg, M, valaccuracy,val_acc.getacc()))
			model.train()
			avgf1=(valconfusion.summary()[0]+valconfusion.summary()[1]+valconfusion.summary()[2])/3.0
			weightnor=0.399
			weightagg=0.257
			weightdrow=0.344
			weightf1=valconfusion.summary()[0]*weightnor+valconfusion.summary()[1]*weightagg+valconfusion.summary()[2]*weightdrow
			print("Epoch: %d, Valloss: %f, sum: %d,  valacc: %f, avgf1: %f, weightf1: %f"%(epoch,  val_losses.avg, val_losses.count, val_acc.getacc(),avgf1,weightf1))
			with open(file="/root/GLMDrivenet/log_motor/val.txt", mode="a+") as f:
				f.write("Epoch: %d,  Valloss: %f, sum: %d,  valacc: %f, avgf1: %f, weightf1: %f\n"%(epoch,val_losses.avg, val_losses.count, val_acc.getacc(),avgf1,weightf1))
			is_best_avgf1=avgf1>best_avgf1
			is_best_weightf1=weightf1>best_weightf1
			is_best = val_acc.getacc() > best_precision
			is_lowest_loss = val_losses.avg < lowest_loss
			best_precision = max(val_acc.getacc(), best_precision)
			lowest_loss = min(val_losses.avg, lowest_loss)
			best_avgf1=max(avgf1,best_avgf1)
			best_weightf1=max(weightf1,best_weightf1)
			with open(file="/root/GLMDrivenet/log_motor/val.txt", mode="a+") as f:
				f.write("Epoch: %d,best_precision: %f,lowest_loss: %f,best_avgf1: %f,best_weightf1: %f"%(epoch, best_precision,lowest_loss,best_avgf1,best_weightf1))
			print('--'*30)
			print("Epoch: %d,best_precision: %f,lowest_loss: %f,best_avgf1: %f,best_weightf1: %f"%(epoch, best_precision,lowest_loss,best_avgf1,best_weightf1))
			print('--' * 30)
		
			save_path = os.path.join(checkpoint_dir,model_name)
			torch.save(model.state_dict(),save_path)
		
			best_path = os.path.join(checkpoint_dir,'best_model.pt')
			if is_best:
				shutil.copyfile(save_path, best_path)
		
			lowest_path = os.path.join(checkpoint_dir, 'lowest_loss.pt')
			if is_lowest_loss:
				shutil.copyfile(save_path, lowest_path)

			best_avgf1_path = os.path.join(checkpoint_dir, 'best_avgf1.pt')
			if is_best_avgf1:
				shutil.copyfile(save_path, best_avgf1_path)
		
			best_weightf1_path = os.path.join(checkpoint_dir, 'best_weightf1.pt')
			if is_best_weightf1:
				shutil.copyfile(save_path, best_weightf1_path)
				

	except Exception as e:
		print(e)
		torch.save(model.state_dict(), "backup"+model_name)
		print("Checkpoint saved and backup.")
	#
	# lossfile.close()
	# lossfile1.close()



def getAVENet(use_cuda):
	model = AVENet1()
	if use_cuda:
		model = model.cuda()

	return model



class TestMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val
		self.count += n

	def getacc(self):
		return (self.sum *100) / self.count


def test(use_cuda=True, batch_size=8, model_name="/root/GLMDrivenet/model_save/lowest_loss.pt"):
	model = getAVENet(use_cuda)
	if os.path.exists(model_name):
		model.load_state_dict(torch.load(model_name))
		print("Loading from previous checkpoint.")

	testdataset = Mydata(img_speed_path='/root/autodl-tmp/motor/test.txt',
	 img_path='/root/autodl-tmp/motor_dataset/test/img/',
	 speed_path='/root/autodl-tmp/motor_dataset/test/speed/')
	testdataloader = DataLoader(testdataset, batch_size=batch_size, shuffle=True, num_workers=2)

	crossEntropy = nn.CrossEntropyLoss()
	print("Loaded dataloader and loss function.")

	test_losses = LossAverageMeter()
	test_acc = TestMeter()
	labels=['Normal','Aggressive','Drowsy']
	testconfusion = testConfusionMatrix(num_classes=3, labels=labels)
	model.eval()
	for sepoch, (img, aud, out) in enumerate(testdataloader):
		out = out.squeeze(1)
		idx = (out != 3).numpy().astype(bool)
		if idx.sum() == 0:
			continue
		# Find the new variables
		img = torch.Tensor(img.numpy()[idx, :, :, :])
		aud = torch.Tensor(aud.numpy()[idx, :, :, :])
		out = torch.LongTensor(out.numpy()[idx])
		img = Variable(img, volatile=True)
		aud = Variable(aud, volatile=True)
		out = Variable(out, volatile=True)
		# print(img.shape, aud.shape, out.shape)
		M = img.shape[0]
		if use_cuda:
			img = img.cuda()
			aud = aud.cuda()
			out = out.cuda()
		with torch.no_grad():
			o, _, _ = model(img, aud)
			valloss = crossEntropy(o, out)
		test_losses.update(valloss.item(), M)
		# Calculate valaccuracy
		o = F.softmax(o, 1)
		_, ind = o.max(1)
		testconfusion.update(ind.to("cpu").numpy(), out.to("cpu").numpy())
		x = (ind.data == out.data).sum() * 1.0
		testaccuracy =x / M
		test_acc.update(x, M)
		if sepoch % 300 == 0:
			print("Sepoch: %d, testloss: %f, batch_size: %d,  testacc: %f, zongacc: %f" % (
			sepoch, test_losses.avg, M,testaccuracy, test_acc.getacc()))
			with open(file="/root/GLMDrivenet/log_motor/test.txt", mode="a+") as f:
				f.write(" Sepoch: %d, testloss: %f, batch_size: %d,  testacc: %f, zongacc: %f\n" % (
				sepoch, test_losses.avg, M, testaccuracy, test_acc.getacc()))
	with open(file="/root/GLMDrivenet/log_motor/test.txt", mode="a+") as f:
		f.write("  testloss: %f, batch_size: %d, sum :%d,  testacc: %f\n" % (test_losses.avg, M,test_acc.count, test_acc.getacc()))
	testconfusion.summary()
	testconfusion.plot()





if __name__ == "__main__":
	cuda = True
	args = parser.parse_args()
	mode = args.mode
	# list_image1=getimage()
	if mode == "demo":
		demo()
	elif mode == "main":
		main(use_cuda=cuda, batch_size=16)
	elif mode == "test":
		test(use_cuda=cuda, batch_size=16)
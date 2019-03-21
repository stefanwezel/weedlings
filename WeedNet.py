import torch
import torchvision
from torch import nn




class WeedNet(torch.nn.Module):
	""" Model of a simple feed forward convolutional neural... """ 
	def __init__(self):
		# uses params, methods of parent class
		super(WeedNet, self).__init__()

		# define sets of layers (convolution, pooling, normalization)
		# convolute (create filter for first convolution (map in_cannel to out_channels))
		self.conv_1 = torch.nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 7) 
		# normalize (reduces the covariation in outputs that get passed to next layer) 
		self.bn1 = nn.BatchNorm2d(16)

		# do the same thing for next set of layers...
		self.conv_2 = torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5) 
		self.bn2 = nn.BatchNorm2d(32)

		# aaaaaand the next set...	
		self.conv_3 = torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5) 
		self.bn3 = nn.BatchNorm2d(64)


		# self.conv_4 = torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 2)
		# self.bn4 = nn.BatchNorm2d(128)


		# self.conv_5 = torch.nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 2)
		# self.bn5 = nn.BatchNorm2d(256)


		# fully connected layer

		self.fc_1 = torch.nn.Linear(in_features = 64, out_features = 32)
		# self.fc_2 = torch.nn.Linear(in_features = 128, out_features = 64)
		self.fc_2 = torch.nn.Linear(in_features = 32, out_features = 12)


	
	def forward(self, x):
		""" define how data is passed from layer to layer. """

		# create relu (delinearization) layer based on output of normalized conv1 layer
		x = torch.nn.functional.relu(self.bn1(self.conv_1(x)))
		# create pooling layer, based on previous relu layer
		x = torch.nn.functional.max_pool2d(x, 4)

		# print(x.size())

		# relu and pool again.. this time for normalized conv2 layer
		x = torch.nn.functional.relu(self.bn2(self.conv_2(x)))
		x = torch.nn.functional.max_pool2d(x, 4)

		# print(x.size())

		# and the same for conv 3
		x = torch.nn.functional.relu(self.bn3(self.conv_3(x)))
		x = torch.nn.functional.max_pool2d(x, 4)


		# print(x.size())
		# x = torch.nn.functional.relu(self.bn4(self.conv_4(x)))
		# x = torch.nn.functional.max_pool2d(x, 4)


		# x = torch.nn.functional.relu(self.bn5(self.conv_5(x)))
		# x = torch.nn.functional.max_pool2d(x, 4)




		# flatten input
		x = x.view(x.size(0), -1)

		# delinearize (relu) first fully connected 
		x = torch.nn.functional.relu(self.fc_1(x))

		# x = torch.nn.functional.relu(self.fc_2(x))

		# call last layer operation
		x = self.fc_2(x)

		# return softmaxed tensor
		return torch.nn.functional.log_softmax(x, dim = 1) # dim = 1?





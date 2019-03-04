import torch
import torchvision



class WeedNet(torch.nn.Module):
	""" Model of a simple feed forward convolutional neural... """ 
	def __init__(self):
		# uses params, methods of parent class
		super(WeedNet, self).__init__()

		# define sets of layers (convolution, pooling, normalization)
		# convolute (create filter for first convolution (map in_cannel to out_channels))
		self.conv_1 = torch.nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 3) 
		# normalize (reduces the covariation in outputs that get passed to next layer) 
		self.bn1 = nn.BatchNorm2d(16)
		# set pooling (moves kernel over layer and chooses max value from each)
		# self.pool = torch.nn.MaxPool2d(4,4) # use the max values of each 4x4 kernel

		# do the same thing for next set of layers...
		self.conv_2 = torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3) 
		self.bn2 = nn.BatchNorm2d(32)
		# aaaaaand the next set...	
		self.conv_3 = torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3) 
		self.bn3 = nn.BatchNorm2d(64)


		# TODO create fully connected layer


		# TODO define forward function

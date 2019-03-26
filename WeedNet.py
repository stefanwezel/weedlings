import torch
import torchvision
from torch import nn
from utils import calculate_probability


class WeedNet(torch.nn.Module):
	""" 
	Model of a feed forward convolutional neural network with twelve outputs, consisting mainly of three
	convolutional layers, with each having additional normalization layers. When forwarding data
	through the model, additional operations, such as max-pooling and delinearization are applied as well.
	At the end, there are two fully connected layers, which eventually produce the classifier's output.
	""" 
	def __init__(self):
		# uses params, methods of parent class
		super(WeedNet, self).__init__()

		# define sets of layers (convolution, pooling, normalization)
		# convolute (create filter for first convolution (map in_cannel to out_channels))
		self.conv_1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 7) 
		# normalize (reduces the covariation in outputs that get passed to next layer) 
		self.bn1 = nn.BatchNorm2d(16)

		# do the same thing for next set of layers...
		self.conv_2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5) 
		self.bn2 = nn.BatchNorm2d(32)

		# aaaaand the next set...	
		self.conv_3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5) 
		self.bn3 = nn.BatchNorm2d(64)

		# fully connected layers
		self.fc_1 = nn.Linear(in_features = 64, out_features = 32)
		self.fc_2 = nn.Linear(in_features = 32, out_features = 12)

		self.last_probabilities = None

	
	def forward(self, x):
		""" Define how data is passed from layer to layer. """

		# create relu (delinearization) layer based on output of normalized conv1 layer
		x = nn.functional.relu(self.bn1(self.conv_1(x)))
		# create pooling layer, based on previous relu layer
		x = nn.functional.max_pool2d(x, 4)

		# relu and pool again.. this time for normalized conv2 layer
		x = nn.functional.relu(self.bn2(self.conv_2(x)))
		x = nn.functional.max_pool2d(x, 4)

		# and the same for conv 3
		x = nn.functional.relu(self.bn3(self.conv_3(x)))
		x = nn.functional.max_pool2d(x, 4)

		# flatten input
		x = x.view(x.size(0), -1)
		# delinearize (relu) first fully connected 
		x = nn.functional.relu(self.fc_1(x))
		
		# call last layer operation
		x = self.fc_2(x)

		self.last_probabilities = calculate_probability(x)
		# return log_softmaxed tensor
		x = nn.functional.log_softmax(x, dim = 1)
		return x


	def load_weights(self, path):
		"""Loads a state dict for a WeedNet"""

		self.load_state_dict(torch.load(path))


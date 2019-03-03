"""
Returns bootstraped, normalized and split data...
"""
import torch
import torchvision
import torchvision.transforms as transforms

from _config import SPLIT_DATA_PATH



# compose a transform which rotates, crops, normalizes and tensorizes an image
transform = transforms.Compose([
	transforms.CenterCrop(150),
	transforms.RandomRotation(360),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])



# create a training data set TODO: turn into function
training_set = torchvision.datasets.ImageFolder(
	root = SPLIT_DATA_PATH + 'train/',
	transform = transform)


training_loader = torch.utils.data.DataLoader(
	training_set,
	batch_size = 4,
	shuffle = True,
	num_workers = 2)






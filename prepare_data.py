"""
Returns bootstraped, normalized and split data...
"""
import torch
import torchvision
import torchvision.transforms as transforms

from _config import SPLIT_DATA_PATH





# compose a transform which rotates, crops, normalizes and tensorizes an image
transform = transforms.Compose([
	transforms.CenterCrop(158),
	transforms.Resize(158),
	transforms.RandomVerticalFlip(0.5),
	transforms.RandomHorizontalFlip(0.5),
	transforms.RandomRotation(360),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])



# create a training data set 
training_set = torchvision.datasets.ImageFolder(
	root = SPLIT_DATA_PATH + 'train/',
	transform = transform)

# load batch of training data set
training_loader = torch.utils.data.DataLoader(
	training_set,
	batch_size = 1,
	shuffle = True,
	num_workers = 2)



# create a training data set 
validation_set = torchvision.datasets.ImageFolder(
	root = SPLIT_DATA_PATH + 'validation/',
	transform = transform)

# load batch of training data set
validation_loader = torch.utils.data.DataLoader(
	validation_set,
	batch_size = 4,
	shuffle = True,
	num_workers = 2)


# create a training data set 
test_set = torchvision.datasets.ImageFolder(
	root = SPLIT_DATA_PATH + 'test/',
	transform = transform)

# load batch of training data set
test_loader = torch.utils.data.DataLoader(
	test_set,
	batch_size = 4,
	shuffle = False,
	num_workers = 2)


# print(test_loader)
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


training_transforms = transforms.Compose([
	transforms.CenterCrop(158),
	transforms.Resize(158),
	transforms.RandomVerticalFlip(0.5),
	transforms.RandomHorizontalFlip(0.5),
	transforms.RandomRotation(360),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])


test_transforms = transforms.Compose([
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




def create_loader(folder, transforms, batch_size = 4):
	dataset = torchvision.datasets.ImageFolder(
		root = SPLIT_DATA_PATH + folder,
		transform = transforms)
	
	loader = torch.utils.data.DataLoader(
		dataset,
		batch_size = batch_size,
		shuffle = True,
		num_workers = 2)

	return loader
# print(test_loader)

def prepare_image(image):
	"""prepares an Image and returns a Tensor with 4 Dimensions and right size"""

	# transforms to the right size and Tensor
	transf = transforms.Compose([
	transforms.Resize(158),
	transforms.ToTensor()
	])
	
	transformed_image = transf(image)

	# adding a dimension to the Tensor for the batch_size = 1
	transformed_image = transformed_image[None,:,:,:]
	return transformed_image
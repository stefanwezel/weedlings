"""
Helper module to visualize data or the models prediction, etc...
"""

import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch
import os
#import seaborn as sns

from prepare_data import training_loader, validation_loader, test_loader, prepare_image
from _config import SPLIT_DATA_PATH, MODEL_PATH
from WeedNet import WeedNet
from utils import predictions, print_dict

def _show_image(image, labels = '', prediction = ''):
	""" Helper function two plot an image ffrom an numpy array. """
	image = image / 2 + 0.5     # unnormalize
	npimg = image.numpy()
	plt.title(labels + '\n' + prediction)
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()



def plot_random_batch(loader, classes,model = None,batch_size = 4):
	""" Picks random images from dataload, unnormalizes them and plots them
	and prints out their labels. """
	
	# pick random files
	dataiter = iter(loader)
	images, labels = dataiter.next()

	label = 'labels:  ' +', '.join('%5s' % classes[labels[j]] for j in range(batch_size))
	prediction = 'no model loaded'
	# print the guess of the potencial model
	if model != None:
		model.eval()
		output = model(images)
		pred = output.argmax(dim = 1, keepdim = True)
		prediction = 'prediction:  ' + ', '.join('%5s' % classes[pred[j]] for j in range(batch_size))
		p = predictions(model.last_probabilities,classes)
		for dic in p:
			print_dict(dic)
	# plot images
	_show_image(torchvision.utils.make_grid(images), label, prediction)
	# print labels
	print(label)
	# print predictions
	print(prediction)



def loss_to_epochs(*args):
	""" plots a lists of Tupels(Points), each arg is a new graph"""
	
	for graph in args:
		plot = list(zip(*graph))
		plt.plot(plot[0],plot[1])
		
	plt.xlabel('epochs')
	plt.ylabel('loss / accuracy')
	plt.show()



def plot_two_graphs(loss, accuracy):
	""" Plots two graphs in one figure, where each has its own y-axis. """
	
	# set colors
	loss_color = '#0082a4' # blueish
	accuracy_color = '#00a474' # greenish

	# format passed data
	loss_zipped = list(zip(*loss))
	accuracy_zipped = list(zip(*accuracy))

	# init figure
	fig, ax1 = plt.subplots()

	# create first y axis and x axis
	ax1.plot(loss_zipped[0], loss_zipped[1], color = loss_color)
	ax1.set_xlabel('epochs (s)')
	ax1.set_xlim(left = 0)
	ax1.set_ylabel('loss', color = loss_color)
	ax1.tick_params(axis = 'y', labelcolor = loss_color)
	# format left y axis
	ax1.set_ylim(bottom = 0)
	vals = ax1.get_yticks()
	ax1.set_yticklabels(['{:,.3}'.format(x) for x in vals])

	# create a second y axis
	ax2 = ax1.twinx()
	ax2.plot(accuracy_zipped[0], accuracy_zipped[1], color = accuracy_color)
	ax2.set_ylabel('accuracy', color = accuracy_color)
	ax2.tick_params(axis = 'y', labelcolor = accuracy_color)
	# format right y axis
	ax2.set_ylim(bottom = 0)
	vals = ax2.get_yticks()
	ax2.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
	
	# show figure
	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.show()


def guess_image(img, classes, model):
	"""plots PIL Image with prediction of the model"""
	model.eval()
	transform_to_PILImage = transforms.ToPILImage()
	img_trans = prepare_image(img)
	output = model(img_trans)
	pred = output.argmax(dim = 1, keepdim = True)
	p = predictions(model.last_probabilities, classes)
	print_dict(p[0])
	prediction = 'Prediction: ' + classes[pred.item()]
	img_trans = img_trans.squeeze(0)
	plt.imshow(transform_to_PILImage(img_trans))
	plt.title(prediction)
	plt.show()
	
# TODO: visualize a random test image and the models prediction for it
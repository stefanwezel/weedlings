"""
Helper module to visualize data or the models prediction, etc...
"""

import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
import os

from prepare_data import training_loader, validation_loader, test_loader
from _config import SPLIT_DATA_PATH, MODEL_PATH
from WeedNet import WeedNet


def _show_image(image, labels = '', prediction = ''):
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
		output = model(images)
		pred = output.argmax(dim = 1, keepdim = True)
		prediction = 'prediction:  ' + ', '.join('%5s' % classes[pred[j]] for j in range(batch_size))

	# plot images
	_show_image(torchvision.utils.make_grid(images), label, prediction)
	# print labels
	print(label)
	# print predictions
	print(prediction)



model = WeedNet()
model.load_state_dict(torch.load(MODEL_PATH + '78.0_percent_accuracy.pt'))
# plot_random_image(training_loader, os.listdir(SPLIT_DATA_PATH + 'train/'))
plot_random_batch(validation_loader, os.listdir(SPLIT_DATA_PATH + 'validation/'), model)
# plot_random_image(test_loader, os.listdir(SPLIT_DATA_PATH + 'test/'))



# TODO: visualize a random test image and the models prediction for it
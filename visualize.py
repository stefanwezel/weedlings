"""
Helper module to visualize data or the models prediction, etc...
"""

import numpy as np
import matplotlib.pyplot as plt
import torchvision
import os

from prepare_data import training_loader, validation_loader, test_loader
from _config import SPLIT_DATA_PATH

def _show_image(image):
    image = image / 2 + 0.5     # unnormalize
    npimg = image.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()



def plot_random_batch(loader, classes, batch_size = 4):
	""" Picks random images from dataload, unnormalizes them and plots them
	and prints out their labels. """
	# pick random files
	dataiter = iter(loader)
	images, labels = dataiter.next()
	# plot images
	_show_image(torchvision.utils.make_grid(images))
	# print labels
	print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))



# plot_random_image(training_loader, os.listdir(SPLIT_DATA_PATH + 'train/'))
plot_random_batch(validation_loader, os.listdir(SPLIT_DATA_PATH + 'validation/'))
# plot_random_image(test_loader, os.listdir(SPLIT_DATA_PATH + 'test/'))



# TODO: visualize a random test image and the models prediction for it
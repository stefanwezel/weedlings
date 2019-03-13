import torch
import torch.optim as optim

from torch import nn
from prepare_data import training_loader, create_loader, transform, validation_loader
from WeedNet import WeedNet
from _config import MODEL_PATH
import copy
import numpy as np
import time
from utils import print_time, progress
import visualize
from test import test

TRAINING_EPOCHS = 4
# init model
WEED_NET = WeedNet()
# criterion is the rule for stpping the algorithm... 
# cross entropy is the average number of bits needed to 
# decide from which one of two  propability functions an event is drawn
CRITERION = nn.CrossEntropyLoss()


def train(*args, model = WEED_NET, criterion = CRITERION, training_epochs = TRAINING_EPOCHS, batch_size = 32, learning_rate = 0.0001):

	# optimizer searches fo a local minimum of in the lossfunction with different input parameters
	#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

	optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 0.001)
	graph_loss = []
	graph_accuracy = [(0,0)]

	best_model = None

	threshhold = 0

	for epoch in range(training_epochs):
		running_loss = 0.0

		training_loader = create_loader('train/', transform, batch_size  = batch_size)
		average_loss = 0
		print('')
		number_of_files = len(training_loader.dataset)
		for i, data in enumerate(training_loader, 0):

			# get input for training
			inputs, labels = data

			# init optimizer with 0
			optimizer.zero_grad()

			# rung data trough net
			outputs = model(inputs)

			# compute loss (compare output to label)
			loss = criterion(outputs, labels)

			# backpropagate loss
			loss.backward()

			# tweak parameters
			optimizer.step()

			# add loss to overall loss
			running_loss += loss.item()

			# pretty print progress
			if i % 10 == 9:  # append a the average of the last 10 losses as point to the loss/epoch graph_loss
				average_loss = running_loss/10
				graph_loss.append((epoch + i/(number_of_files/batch_size), average_loss))
				running_loss = 0.0

			# Progress bar
			progress(i, number_of_files/batch_size, epoch + 1, '{}/{:.0f} Loss: {:.2f}'.format(i, number_of_files/batch_size, average_loss))

		# Validate the result of the epoch
		test_loss, correct, dataset_size, accuracy_percent = test(validation_loader, model)
		graph_accuracy.append((epoch + 1, accuracy_percent/100))
		if accuracy_percent > threshhold:
			best_model = copy.deepcopy(model)

	model_name = '{}epochs_{}learingrate_{}batchsize.pt'.format(training_epochs, learning_rate, batch_size)
	torch.save(best_model.state_dict(), MODEL_PATH + model_name)

	print("\nmodel: " + model_name + " has been saved.")
	return best_model, (graph_loss, graph_accuracy)
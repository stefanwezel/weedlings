import os
import torch

from torch import nn
from _config import MODEL_PATH, SPLIT_DATA_PATH
from train import train
from test import test
from WeedNet import WeedNet
from prepare_data import create_loader, test_transforms, test_loader, validation_loader
from visualize import plot_random_batch, loss_to_epochs
from datetime import datetime


# init model
WEED_NET = WeedNet()

# set hyperparameters
TRAINING_EPOCHS = 1
NUMBER_OF_TESTS = 3
LEARNING_RATE = 0.0007
BATCH_SIZE = 32

# start timer
start = datetime.now()

# train
model, graph = train(
	model = WEED_NET,
	criterion = nn.CrossEntropyLoss(),
	training_epochs = TRAINING_EPOCHS,
	learning_rate = LEARNING_RATE,
	batch_size = BATCH_SIZE)

# test
for i in range(NUMBER_OF_TESTS):
	test(test_loader, model)


print("\nOverall training and testing time: " + str(datetime.now() - start))

# plot loss over epochs
loss_to_epochs(graph[0], graph[1], graph[2])



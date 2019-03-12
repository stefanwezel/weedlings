import os
import torch

from _config import MODEL_PATH, SPLIT_DATA_PATH
from train import train
from test import test
from WeedNet import WeedNet
from prepare_data import create_loader, test_transforms, test_loader, validation_loader
from visualize import plot_random_batch, loss_to_epochs
from datetime import datetime




NUMBER_OF_TESTS = 3

# start timer
start = datetime.now()

# train
model, graph = train(training_epochs = 3, learning_rate = 0.001)

# test
for i in range(NUMBER_OF_TESTS):
	test(test_loader, model)


print("\nOverall training and testing time: " + str(datetime.now()-start))

# plot loss over epochs
loss_to_epochs(graph)



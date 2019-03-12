import os
import torch
from _config import MODEL_PATH, SPLIT_DATA_PATH
from train import train
from test import test
from WeedNet import WeedNet
from prepare_data import create_loader, test_transforms, test_loader, validation_loader
import time
from utils import print_time
from visualize import plot_random_batch, loss_to_epochs

NUMBER_OF_TESTS = 3

#test_loader = create_loader('test/', test_transforms)

model, graph = train(training_epochs = 1)

for i in range(NUMBER_OF_TESTS):
	test(test_loader, model)

t = time.clock()

print_time(t, "train and test")

loss_to_epochs(graph)


# model = WeedNet()
# model.load_state_dict(torch.load(MODEL_PATH + '78.0_percent_accuracy.pt'))

# for x in range(NUMBER_OF_TESTS):
# 	plot_random_batch(validation_loader, os.listdir(SPLIT_DATA_PATH + 'validation/'), model)


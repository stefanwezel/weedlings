import os
import torch
from _config import MODEL_PATH, SPLIT_DATA_PATH
from train import train
from test import test
from WeedNet import WeedNet
from prepare_data import create_loader, test_transforms, test_loader, validation_loader
import time
from utils import print_time
from visualize import plot_random_batch

NUMBER_OF_TESTS = 3

# test_loader = create_loader('test/', test_transforms)

model = train(training_epochs = 1)

for i in range(NUMBER_OF_TESTS):
	test(test_loader, model)

t = time.clock()

print_time(t, "train and test")


plot_random_batch(validation_loader, os.listdir(SPLIT_DATA_PATH + 'validation/'), model)


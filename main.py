import torch
from _config import MODEL_PATH
from train import train
from test import test
from WeedNet import WeedNet
from prepare_data import create_loader, test_transforms, test_loader

NUMBER_OF_TESTS = 3

# test_loader = create_loader('test/', test_transforms)


model = train(training_epochs = 1)

for i in range(NUMBER_OF_TESTS):
	test(test_loader, model)
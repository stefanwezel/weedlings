import torch
from _config import MODEL_PATH
from train import train
from test import test
from WeedNet import WeedNet

NUMBER_OF_TESTS = 5

model = train()

for i in range(NUMBER_OF_TESTS):
	test(model = model)
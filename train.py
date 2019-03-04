import torch
import torch.optim as optim

from torch import nn
from prepare_data import training_loader
from WeedNet import WeedNet

TRAINING_EPOCHS = 4





# init model
weed_net = WeedNet()
# criterion is the rule for stpping the algorithm... 
 # cross entropy is the average number of bits needed to 
 # decide from which one of two  propability functions an event is drawn
criterion = nn.CrossEntropyLoss()
# optimizer searches fo a local minimum of in the lossfunction with different input parameters
optimizer = optim.SGD(weed_net.parameters(), lr=0.001, momentum=0.9)






for epoch in range(TRAINING_EPOCHS):
	running_loss = 0.0
	for i, data in enumerate(training_loader, 0):
		# get input for training
		inputs, labels = data
		# init optimizer with 0
		optimizer.zero_grad()
		# rung data trough net
		outputs = weed_net(inputs)
		# compute loss (compare output to label)
		loss = criterion(outputs, labels)
		# backpropagate loss
		loss.backward()
		# tweak parameters
		optimizer.step()
		# add loss to overall loss
		running_loss += loss.item()
		# pretty print progress
		if i % 200 == 199:    # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / 200))
			running_loss = 0.0

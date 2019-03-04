import torch
import torch.nn as nn
import torch.nn.functional as F

from WeedNet import WeedNet
from prepare_data import test_loader

weed_net = WeedNet()

device = torch.device("cpu")

def test(*args, weed_net = weed_net, device = device, test_loader = test_loader):
	weed_net.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = weed_net(data)
			test_loss += F.nll_loss(output, target, reduction = 'sum').item()
			pred = output.argmax(dim = 1, keepdim = True)
			correct += pred.eq(target.view_as(pred)).sum().item()

		test_loss /= len(test_loader.dataset)

		print('\nTest set: Average loss:  {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100 * correct/len(test_loader.dataset)))


test(weed_net = weed_net, device = device, test_loader = test_loader)

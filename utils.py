import time
import sys

def print_time(t, text):
	hours = t/3600
	t = t % 3600
	minutes = t/60
	t = t % 60

	print("\ntime to " + text +": {:.0f} hours, {:.0f} minutes, {:.2f} seconds".format(hours, minutes, t))



label_dict = {	0: 'black_grass',
				1: 'charlock',
				2: 'cleavers',
				3: 'common_chickenweed',
				4: 'common_wheat',
				5: 'fat_hen',
				6: 'loose_silky_bent',
				7: 'maize',
				8: 'scentless_mayweed',
				9: 'shephard\'s_purse',
				10:'small_flowered_cranesbill',
				11:'sugar_beet'}





def progress(count, total, epoch, suffix = ''):
	""" Writes a progress bar to console. """
	bar_len = 40
	filled_len = int(round(bar_len * count / float(total)))

	percents = round(100.0 * count / float(total), 1)
	bar = '=' * filled_len + '-' * (bar_len - filled_len)
	# sys.stdout.write('epoch %s:\r' %(epoch))
	sys.stdout.write('epoch %s: [%s] %s%s ...%s\r' % (epoch, bar, percents, '%', suffix))
	# sys.stdout.write('average_loss: %s\r' % (average_loss))
	sys.stdout.flush()  # As suggested by Rom Ruben

def predictions(lis, classes):
	result = []
	counter = 0
	
	for i in range(len(lis)):
		dic = {}
		for j in range(len(lis[0])):
			dic[classes[j]] = lis[i][j]*100
			
		result.append(dic)

	return result

def print_dict(dic):
	result = ''
	for key in dic.keys():
		result += key + ': {:.2f}\n'.format(dic[key])

	print(result)

def calculate_probability(Tensor):
	result = []
	for i in range(Tensor.size()[0]):
		tensor_result = []
		s = 0
		for j in range(Tensor.size()[1]):
			next_number = Tensor[i][j].item()
			if next_number > 0:
				s += next_number

		for j in range(Tensor.size()[1]):
			next_number = Tensor[i][j].item()
			if next_number > 0:
				tensor_result.append(next_number/s)
			else:
				tensor_result.append(0)

		result.append(tensor_result)
	return result


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





def progress(count, total, epoch, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    # sys.stdout.write('epoch %s:\r' %(epoch))
    sys.stdout.write('epoch %s: [%s] %s%s ...%s\r' % (epoch, bar, percents, '%', suffix))
    # sys.stdout.write('average_loss: %s\r' % (average_loss))
    sys.stdout.flush()  # As suggested by Rom Ruben
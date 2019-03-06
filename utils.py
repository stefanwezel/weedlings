import time

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
				5: 'fat_han',
				6: 'loose_silky_bent',
				7: 'maize',
				8: 'scentless_mayweed',
				9: 'shepard\'s_purse',
				10:'small_flowered_cranesbill',
				11:'sugar_beet'}
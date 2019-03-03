"""
Based on a root folder containing labeled data, creates a train, validation and test directory.
"""

import os
import sys
import json
import random

from shutil import rmtree, copyfile # copyfile(src, dst)
from _config import PATH, DATA_PATH, RAW_DATA_PATH, SPLIT_DATA_PATH



def create_directories(path, directories = ['train/', 'validation/', 'test/']):
	""" 
	Creates given directories in given path. 
	Removes existing ones. Defaults are: train, validation, test.
	"""
	try:
		for directory in directories:
			if os.path.isdir(path + directory):
				rmtree(path + directory)
				print("Removing existing directory: " + directory)

			os.mkdir(path + directory)
			for label in os.listdir(RAW_DATA_PATH):
				# create subfolders for types of weed and use pythonic naming conventions
				os.mkdir(path + directory + label.replace(" ", "_").replace("-", "_").lower() + "/")
	except FileNotFoundError as e:
		print(e)
		print("\nSomething went wrong. Make sure the path you specified exists...")
		sys.exit("\nExit...")


	print("Folders ", [str(directory) for directory in directories], " created succesfully...")


create_directories(SPLIT_DATA_PATH)



# load image dict from json file
try:
	with open(PATH + 'image_dict.json') as json_dict:
		image_dict = json.load(json_dict)
		
		# check if dict is empty
		if len(image_dict.keys()) == 0:
			raise IOError
		
except (FileNotFoundError, IOError) as e:
	print(str(e))
	if "No such file" in str(e):
		print("The dictionary could not be found. Make sure it exists in the given path.")
	else:
		print("The dictionary seems to be empty...")
	
	sys.exit("\nExit...")



# copy the files from source to target direcories using the image_dict
for weed in image_dict.keys():
	for sample in image_dict[weed].keys():
		# by 2/3 chance: -> copy to training
		random_sample = random.random()
		sample_new_file_name = sample.replace(" ", "_").lower()
		if random_sample <= 0.66:
			copyfile(
				image_dict[weed][sample],
				SPLIT_DATA_PATH + 'train/' + str(weed).replace(" ", "_").replace("-", "_").lower() + '/' + sample_new_file_name + '.png'
				)
			print("copied to train")
		# by 1/6 chance:-> copy to validation
		elif (random_sample > 0.66) and (random_sample <= 0.83):
			copyfile(
				image_dict[weed][sample],
				SPLIT_DATA_PATH + 'validation/' + str(weed).replace(" ", "_").replace("-", "_").lower() + '/' + sample_new_file_name + '.png'
				)
			print("copied to validation")
		# by 1/6 chance: -> copy to test
		else:
			copyfile(
				image_dict[weed][sample],
				SPLIT_DATA_PATH + 'test/' + str(weed).replace(" ", "_").replace("-", "_").lower() + '/' + sample_new_file_name + '.png'
				)
			print("copied to test")



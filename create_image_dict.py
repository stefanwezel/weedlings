"""
Returns a dict with form:
{
Plant_1(label): {image_1_name: path_to_image_1}, {image_2_name: path_to_image_2}, ..., {image_n_name: path_to_image_n},
Plant_2(label): {image_1_name: path_to_image_1}, {image_2_name: path_to_image_2}, ..., {image_n_name: path_to_image_n},
...,
Plant_n(label): {image_1_name: path_to_image_1}, {image_2_name: path_to_image_2}, ..., {image_n_name: path_to_image_n}
}
"""
import os
import json

from _config import PATH, DATA_PATH


# get the subfolders of the directory where data is stored
try:
	plants = os.listdir(DATA_PATH)
	if len(plants) == 0:
		print("\nThe folder you specified in _conf.py seems empty...")
		raise FileNotFoundError
		raise
except FileNotFoundError:
	print("\nSomething went wrong... make sure the path you specified in _conf.py is correct...\n")


# init empty dict
plant_dict = {}

# populate the dict
for plant_type in plants:
	plant_dict[plant_type] = {}
	plant_imgs = os.listdir(DATA_PATH + plant_type)
	for plant_img in plant_imgs:
		plant_dict[plant_type][plant_type + "_" + plant_img.replace(".png", "")] = DATA_PATH + plant_img 


# write the created dict to a json file
with open(PATH + 'image_dict.json', 'w') as outfile:
	json.dump(plant_dict, outfile)


print("\nImage dict created succefully at " + PATH + "image_dict.json ...\n")

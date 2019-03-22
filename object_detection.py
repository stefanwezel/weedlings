# WeedNet imports
from _config import MODEL_PATH, SPLIT_DATA_PATH
from WeedNet import WeedNet
from test import test
from prepare_data import create_loader, test_transforms, test_loader, validation_loader

# libraries
import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

# image libraries
from PIL import Image
import cv2


# definitions
MODEL_NAME = '300epochs_0.0007learingrate_32batchsize.pt'
conf_threshhold = 0.8
nms_threshhold = 0.4
img_size = 128

# Load model
model = WeedNet()
model.load_weights(MODEL_PATH + MODEL_NAME)
model.eval()

Tensor = torch.FloatTensor # ? Tensor = torch.cuda.FloatTensor?

def detect_image(img):
	# scale and pad image
	ratio = min(img_size/img.size[0], img_size/img.size[1])
	imw = round(img.size[0] * ratio)
	imh = round(img.size[1] * ratio)
	img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
		transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
						(128,128,128)),
		transforms.ToTensor(),
		])

	# convert image to Tensor
	image_tensor = img_transforms(img).float()
	image_tensor = image_tensor.unsqueeze_(0)
	input_img = Variable(image_tensor.type(Tensor))
	with torch.no_grad():
		detections = model(input_img)


# i tried my best but i dont succed...
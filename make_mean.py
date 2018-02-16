# make mean image from a list of images

import sys
import os
import numpy as np
import cv2

def read_img(path, crop_size):
	img = cv2.imread(path).astype(np.float32)
	width = img.shape[0]
	height = img.shape[1]
	if(width>height):
          scale = float(crop_size)/float(height)
          img = np.array(cv2.resize(np.array(img),(int(width * scale + 1), crop_size))).astype(np.float32)
        else:
          scale = float(crop_size)/float(width)
          img = np.array(cv2.resize(np.array(img),(crop_size, int(height * scale + 1)))).astype(np.float32)
        crop_x = int((img.shape[0] - crop_size)/2)
        crop_y = int((img.shape[1] - crop_size)/2)
        img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:]
	return img

def read_list(path):
	with open(path) as f:
		files = f.readlines()
	return [f.split()[0] for f in files]
	
def calc_mean_img(list_path, crop_size=112):
	img_files = read_list(list_path)
	number_of_image = len(img_files)
	img_accu = np.zeros((crop_size, crop_size, 3))
	for img_file in img_files:
		img_accu += read_img(img_file, crop_size)
	return img_accu / number_of_image


if __name__=="__main__":
	mean_img = calc_mean_img(sys.argv[1])
	np.save("liris_mean.npy", mean_img)

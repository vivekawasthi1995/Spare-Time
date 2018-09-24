import numpy as np
import cv2
import numpy as np
import os

image_height = 64
image_width = 64
image_channel = 1

INPUT_SHAPE = ( image_height, image_width, image_channel )

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def load_image( dir_path, image_path ):
    #return rgb2gray(plt.imread( data_path))
    return cv2.imread( os.path.join(data_path) )

def batch_generator( batch_size, data_dir, no_of_output_cateogory, X_data, Y_data ):

	images = np.zeros( [batch_generator, INPUT_SHAPE] )
	labels = np.zeros( [batch_generator, no_of_output_cateogory])

	while True:
		i = 0
		for index in np.random.permutation( X_data.shape[0]):
			images[i] = load_image( data_dir, X_data[index])
			labels[i] = Y_data[ index]

			i = i+1

			if i == batch_size:
				break

	yield images, labels


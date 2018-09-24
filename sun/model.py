import numpy as np
import pandas as pd
import os
import cv2

import load_data
import data_util
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Lambda, Input
import keras as k
from keras.layers.core import Activation
from keras import optimizers
from sklearn.model_selection import train_test_split


INPUT_SHAPE = ( 64, 64, 3)

path_train_txt = r'/home/cloud/Desktop/oneShot1/data/DatasetA_train_20180813/train.txt'
path_label_txt = r'/home/cloud/Desktop/oneShot1/data/DatasetA_train_20180813/label_list.txt'
image_path = r"/home/cloud/Desktop/oneShot1/data/DatasetA_train_20180813/train"




batch_size = 64
learning_rate = 0.001

X,Y, mappedCick = data_preprocessing1( path_train_txt, path_label_txt)

X_train, X_test, Y_train, Y_test = train_test_split( X, Y)

def build_model( self):

	model = Sequential()
	model.add( Lambda( lambda x: x/127.5-1.0, input_shape = INPUT_SHAPE))
	model.add( Conv2D( 64, [8,8], strides = (1,1), padding = 'valid', activation = 'elu'))
	model.add( MaxPool2D( pool_size = (2,2), strides = (1,1), padding = 'valid'))
	model.add( Conv2D(128, [7,7], strides= (1,1), padding = 'valid'))
	model.add( MaxPool2D())
	model.add( Conv2D( 128, [4, 4], strides = (1,1), activation = 'elu') )
	model.add( MaxPool2D())
	model.add( Dense( 4096))
	model.add( Dense(1024))
	model.add( Dense(231))
	model.add( Activation = 'sigmoid')
	model.add_sumary()
	return model

def train_model( model, batch_size, image_path, X_train, Y_train):
	
	model.compile( loss = 'mean_squared_error', optimizer = Adam( lr = learning_rate))
	model.fit_generater( batch_generator( batch_size, image_path, 231, X_train, Y_train),
		steps_per_epoch = len( X_train)/batch_size, epochs = 10, verbose = 2)




def main():
	model = build_model()
	train_model( model, batch_size, image_path,X_train, Y_train)




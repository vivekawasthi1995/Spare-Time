
# coding: utf-8

# In[399]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Lambda, Input
import keras as k
from keras.layers.core import Activation
from keras import optimizers
from sklearn.model_selection import train_test_split


# In[319]:


import numpy as np
import matplotlib.image as plt
import cv2

image_height = 64
image_width = 64
image_channel = 1

import numpy as np
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def load_image( data_path ):
    #return rgb2gray(plt.imread( data_path))
    return cv2.imread( data_path )


# In[375]:


import os
X_train = []
Y_train = []
with open(r'C:\Users\SRI-D\Desktop\one shot\data\DatasetA_train_20180813\train.txt','r') as f:
    #X_train, Y_train = [x.strip().split('\t') for x in f]
    for x in f:
        x,y = x.strip().split('\t')
        X_train.append(x)
        Y_train.append(y)
    
category = []
label = []
dicMap = {}
with open(r'C:\Users\SRI-D\Desktop\one shot\data\DatasetA_train_20180813\label_list.txt','r') as f:
    for x in f:
        x,y = x.strip().split('\t')
        category.append( y)
        label.append(x)
idd = 0
for x in label:
    dicMap[x]= idd
    idd = idd + 1

y_train = []
#print ( dicMap)
for x in Y_train:
    y_train.append( dicMap[x])


# In[343]:


print ( len(X_train))


# In[376]:


path = r"C:\Users\SRI-D\Desktop\one shot\data\DatasetA_train_20180813\train"
x_train = np.ndarray([3,64,64,3])
x_test = np.ndarray([3,64,64,3])
X_train, X_test, Y_train, y_test = train_test_split( X_train, y_train, test_size = 0.2)
x_train,y_train = data_generate( path, X_train[:100], Y_train[:100] )
x_test, y_test = data_generate( path, X_test[:20], y_test[:20])


# In[348]:


color_channel = batch_size = 3
def data_generate( path, X_train, Y_train):
    
    image = np.zeros( [ len( X_train), image_height, image_width, color_channel] )
    label = np.zeros( len( Y_train) )
    
    i= 0
    for index in np.random.permutation( len( X_train)):
        
        image[i] = load_image( os.path.join( path, X_train[index]))
        #print ( type( image[i]))
        label[i] = y_train[ index ]
        
        i+=1
        
        if i == batch_size :
            break
    print ( type(image))
            
    return image,label    


# In[404]:


batch_size = 64
INPUT_SHAPE = (64, 64,3)
#input_shape = Input( input_shape)
#input_shape = ( batch_size, image_height, image_width)

model = Sequential()
#model = model( input_shape )
#model.add( Input((64, 64 ,1)))
model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
model.add( Conv2D( 64, [8,8], strides = (1,1), padding = 'valid',  activation = 'elu', input_shape = input_shape ))
model.add( MaxPool2D( pool_size = (2,2), strides = (1,1), padding = 'valid' ))
model.add( Conv2D( 128, [7,7], strides = (1,1), padding = 'valid',  activation = 'elu', input_shape = input_shape ))
model.add( MaxPool2D())
model.add( Conv2D( 128, [4, 4], strides = (1,1), activation = 'elu') )
model.add( MaxPool2D())
model.add( Conv2D( 256, [4, 4], strides = (1,1), activation = 'elu') )
model.add( Dense( 4096))
model.add( Dense(230))
#model.add( Flatten())
model.add( Dense(230))
model.add( Flatten(230))
#model.add( Dense(230))
model.add(Activation('sigmoid'))
model.summary()


# In[401]:


adam = optimizers.Adam( 0.00006 )
model.compile( loss = "mean_squared_error",  optimizer = adam )


# In[402]:


y_train = k.utils.to_categorical(y_train, 230)
y_test = k.utils.to_categorical(y_test, 230)


# In[403]:


model.fit( x_train, y_train, batch_size = 3, epochs = 10, verbose =2 , validation_data = (x_test, y_test))


# In[262]:


def batch_generator( batch_size, data_path , X_train, Y_train):
    
    #X = np.zeros([])
    image = np.zeros( [ batch_size, image_height, image_width,3] )
    label = np.zeros( [batch_size])
    
    while True:
        i = 0
        for index in np.random.permutation( len( X_train) ):
            
            image[i] = load_image( os.path.join( data_path, X_train[index]))
            #print ( image[i])
            label[i] = y_train[ index ]
            
            i+=1
            if i == batch_size :
                break
    return images,label


# In[232]:


#X_train, Y_train, X_test, y_test = train_test_split( X_train, y_train)


# In[235]:


model.fit_generator( batch_generator( batch_size, path, X_train, Y_train),
                     steps_per_epoch = len( X_train / batch_size),epochs = 1000,
                     verbose = 2,
                     validation_data = batch_generator( batch_size, path, X_test, Y_test)
                   )


# In[ ]:


print ( X_train.shape)


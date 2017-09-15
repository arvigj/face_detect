'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from keras.applications.vgg19 import VGG19

import matplotlib.pyplot as plt

batch_size = 128
epochs = 1

# input image dimensions

# the data, shuffled and split between train and test sets
x_train = np.load("x_train.npy").astype(np.float32)
y_train = np.ceil(np.load("y_train.npy")).astype(np.float32)

x_train = x_train.reshape([x_train.shape[0],65,65,1])
x_train = np.repeat(x_train,3,axis=3)


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(y_train)

# convert class vectors to binary class matrices

base_model = VGG19(weights="imagenet",include_top=False,pooling="max",input_shape=(65,65,3))
for i in base_model.layers[:-5]:
    i.trainable=False
x = base_model.output
x = Dropout(0.5)(x)
x = Dense(1, activation = 'sigmoid')(x)

model = Model(input=base_model.input, output=x)
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.001,momentum=0.8),
              metrics=['accuracy'])
print(model.summary())

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.1,
          shuffle=True)
model.save("model.h5")

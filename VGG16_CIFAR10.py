import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import ZeroPadding2D
from tensorflow.python.keras.layers import Activation, Dropout
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical



from tensorflow.keras.datasets import cifar10
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

X_train = X_train.reshape(50000, 32, 32, 3) #reshaping the data for the CNN model 
X_test = X_test.reshape(10000, 32, 32, 3)


Y_train = to_categorical(Y_train, 10)      # one hot encoding of traing data label 
Y_test = to_categorical(Y_test, 10)        # one hot encoding of test data label 
     
X_train = X_train/255                      #Normalization of training and test data
X_test = X_test/255


model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu', input_shape = (32, 32, 3)))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))


model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))


model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=256, activation='relu'))
model.add(Dense(units=512, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.summary()

#model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
model.compile(keras.optimizers.Adam(learning_rate=0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
#model.compile(RMSprop(lr = 0.1), loss = 'categorical_crossentropy', metrics = ['accuracy'])


history = model.fit(X_train, Y_train, epochs = 10, validation_split= 0.1, batch_size = 128, verbose = 2, shuffle= 1)


scores = model.evaluate(X_test, Y_test, verbose=0)                     #testing the model with test data
#print(scores)
print("score = ", scores[0])
print("accuracy =  ", scores[1])


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss_training', 'loss_validation'])
plt.title('Loss')
plt.xlabel('epoch')

#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.legend(['accuracy_training','accuracy_validation'])
#plt.title('Accuracy')
#plt.xlabel('epoch')



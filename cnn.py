import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# train data classes
labels = {
        0 : 'casseteplayer',
        1 : 'chainsaw',
        2 : 'church',
        3 : 'englishspringer',
        4 : 'frenchhorn',
        5 : 'garbagetruck',
        6 : 'gaspump',
        7 : 'golfball',
        8 : 'parachute',
        9 : 'tench'
    }

img_height, img_width = 150, 150 # height and width with which we will work
input_shape = (img_height, img_width, 3) # input dimension to the network if you had grayscale images then 3 change to 1
BATCH_SIZE = 12 #how many pictures to train at once, it affects the speed of training and difficulty, the more demanding the training

## data reading
# initialization of the datagenerator which reads the data and defining preprocessing operations to be performed
# you don't even have to use these but I would recommend the normalization

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, #normalization
        shear_range=0.2,
        zoom_range=0.2, # random zoom with a probability of 0.2
        horizontal_flip=True)

# here we will read and count the classes and preprocessing will be performed
train_generator = train_datagen.flow_from_directory(
        'exam_data/train',
        target_size=(img_height,img_width), # target dimension
        batch_size=BATCH_SIZE,
        class_mode='categorical', # change classes to numbers
        shuffle=True) # shuffle dataset

## composing a model
model = Sequential()
# first conv layer with 6 filters having size 5x5, sigmoid activation function and input size
model.add(Conv2D(6, kernel_size=(5, 5),
                activation='sigmoid',
                input_shape=input_shape))
# pooling layer with a size of 2x2
# further as for the first layer

model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Conv2D(12, kernel_size=(5, 5), input_shape=input_shape,
                activation='sigmoid'))
model.add(AveragePooling2D(pool_size=(2, 2)))

# since this model is fully interconnected convolutional network and at the end there is a classic neural network this is necessary
# flatten n dimensional field into 1D
model.add(Flatten())

# output layer with 10 neurons for 10 classes
model.add(Dense(10, activation='sigmoid'))

# list the model summary
model.summary()

# compile the model and define an error function, an optimizer that changes weights and metrics to calculate success
model.compile(loss=mean_squared_error,
            optimizer=Adam(),
            metrics=['accuracy'])

EPOCHS = 1 # number of epochs

## training
model.fit(train_generator,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2)

## prediction
# we go through the validation set and perform a prediction, which it immediately writes to the result.csv
for file in os.listdir("exam_data/val"):
    img = tf.keras.preprocessing.image.load_img("exam_data/val/{}".format(file),target_size=(img_height,img_width))
    img = tf.keras.preprocessing.image.img_to_array(img) 
    img = img / 255.0 # normalzation
    img = np.array([img])
    pred = labels[np.argmax(model.predict(img))] #prediction

    with open("result.csv", "a") as f:
        f.write("{},{}\n".format(file, pred))

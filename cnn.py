import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# train data classes
labels = {
    0: 'cassetteplayer',
    1: 'chainsaw',
    2: 'church',
    3: 'englishspringer',
    4: 'frenchhorn',
    5: 'garbagetruck',
    6: 'gaspump',
    7: 'golfball',
    8: 'parachute',
    9: 'tench'
}

img_height, img_width = 150, 150  # height and width with which we will work
input_shape = (img_height, img_width, 3)  # input dimension to the network
BATCH_SIZE = 32  # how many pictures to train at once

# Data augmentation and normalization for training
train_datagen = ImageDataGenerator(
    rescale=1./255,  # normalization
    shear_range=0.2,
    zoom_range=0.2,  # random zoom
    horizontal_flip=True)

# Read and preprocess the data
train_generator = train_datagen.flow_from_directory(
    'exam_data/train',
    target_size=(img_height, img_width),
    batch_size=BATCH_SIZE,
    class_mode='categorical',  # change classes to numbers
    shuffle=True)  # shuffle dataset

# Compose the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))  # Output layer with softmax activation for classification

# Compile the model
model.compile(optimizer=Adam(), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
EPOCHS = 10  # Adjust the number of epochs according to the need
model.fit(train_generator,
          steps_per_epoch=len(train_generator),
          epochs=EPOCHS,
          verbose=1)

# Save the model
model.save('model.h5')

# Prediction on a single file (example)
def predict_image(filename):
    img = tf.keras.preprocessing.image.load_img(filename, target_size=(img_height, img_width))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.

    prediction = model.predict(img)
    return labels[np.argmax(prediction)]

# Loop through validation set to make predictions
for file in os.listdir("exam_data/val"):
    prediction = predict_image(os.path.join("exam_data/val", file))
    print(f'Prediction for {file}: {prediction}')
    
# Write predictions to result.csv
with open("result.csv", "w") as f:
    for file in os.listdir("exam_data/val"):
        prediction = predict_image(os.path.join("exam_data/val", file))
        f.write("{},{}\n".format(file, prediction))

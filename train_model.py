import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def load_fer2013():
    # Load the FER2013 dataset
    data = pd.read_csv('fer2013.csv')
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    
    # Convert pixels to images
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        faces.append(face.astype('float32'))
    
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)
    
    # Get emotions
    emotions = pd.get_dummies(data['emotion']).values
    
    return faces, emotions

def create_model():
    model = Sequential()
    
    # First Convolutional Layer
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Second Convolutional Layer
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    
    return model

def main():
    print("Loading dataset...")
    faces, emotions = load_fer2013()
    
    # Normalize the data
    faces = faces / 255.0
    
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, random_state=42)
    
    # Create and compile the model
    model = create_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Train the model
    print("Training model...")
    model.fit(
        datagen.flow(x_train, y_train, batch_size=32),
        steps_per_epoch=int(len(x_train) / 32),
        epochs=10,
        validation_data=(x_test, y_test)
    )
    
    # Save the model
    model.save('emotion_model.h5')
    print("Model saved as 'emotion_model.h5'")

if __name__ == "__main__":
    main() 
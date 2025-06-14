import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import scipy


import numpy as np#set up directories
train_dir = r"C:\Users\archi\OneDrive\Desktop\edunet\train"
valid_dir = r"C:\Users\archi\OneDrive\Desktop\edunet\valid"
test_dir = r"C:\Users\archi\OneDrive\Desktop\edunet\test"

# Set up image data generators
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directories
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(64, 64), batch_size=32, class_mode='binary')
valid_generator = valid_datagen.flow_from_directory(valid_dir, target_size=(64, 64), batch_size=32, class_mode='binary')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(64, 64), batch_size=32, class_mode='binary')


#building s simple cnn model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid') #binary classification: wild fire or no wild fire
])

#compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#train model
history=model.fit(train_generator, validation_data=valid_generator,epochs=10,verbose=1)
model.save("fire_detection_model.h5")
print("Model saved as fire_detection_model.h5")
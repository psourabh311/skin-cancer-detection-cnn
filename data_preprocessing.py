#                                          1. Load & Preprocess Data

import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#DATASET DIRECTORIES
BASE_DIR = "/kaggle/input/melanoma-cancer-dataset/"  # Change based on your dataset location
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")

#IMAGE SIZE AND BATCH SIZE
IMG_SIZE = (224,224)  # OPTIMAL SIZE FOR EVERY IMAGE PRESENT IN TEST AND TRAIN DATA
BATCH_SIZE = 32

#DATA AUGMENTATION FOR TRAINING
#NOW IN IMAGE PREPROCESSING KERAS PROVIDES YOU A CLASS THAT IS IMAGEDATAGENERATOR WHICH GIVES YOU TWO THINGS IMAGE KO EXISINTG IMAGE SE DATA GENERATE KARNA AND IMAGE KO PREPROCESS KARNA
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True

# AAP EXISTING DATA SE IMAGE KO GENERATE KARTE HO JISSE YE FAAYDA HAI KI AAPKO JO MODEL HOTA HAI WO OVERFITTING SE BACH JAATA HAI
)
#RESCALING ONLY FOR TESTING
test_datagen = ImageDataGenerator(rescale=1./255)


# Load Training Data
train_generator = train_datagen.flow_from_directory(
    directory=TRAIN_DIR, 
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"  # Binary classification: Benign vs Malignant
)

# Load Testing Data
test_generator = test_datagen.flow_from_directory(
    directory=TEST_DIR, 
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# Print class labels
print("Class Mapping:", train_generator.class_indices)



#                                               2. EXPLORATORY DATA ANALYSIS (EDA)

#DISPLAY SOME SAMPLE IMAGES FROM DATASET
def plot_images(generator):   # plot_images jiske paas generator aaraha hai yaani ki dataset aara hai
    images, labels = next(generator) #use next method to find the images and labels while passing generator
    plt.figure(figsize=(10,10)) # 10inch height and 10 inch width
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i])
        plt.title("Cancer" if labels[i] == 1 else "Non-Cancer")
        plt.axis("off")
    plt.show()

# Plot Training Images
plot_images(train_generator)

#                                                3. BUILD CNN MODEL

from tensorflow.keras import layers, models

# Build the CNN model
model = models.Sequential([
    # IN CNN we have 2 parts (1. feature extractor it's like eyes: see and understand) and (classification its like the brain that decides what it is based on what you saw)
    
 # 1st Convolution Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    
    # 2nd Convolution Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # 3rd Convolution Block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Classification [[20,20,30,30]] [20,20,30,30]
    
    # Flattening Layer
    layers.Flatten(),
    
    # Fully Connected Layers
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Dropout to reduce overfitting
    #2 OR 3 OR MULTICLASS CLASSIFICATION SO USE SOFTMAX AND IF IT IS BINARY USE SIGMOID
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam',  # Optimizer for faster convergence
              loss='binary_crossentropy',  # Binary classification loss and FOR BINARY CLASS CLASSIFICATION WE USE BINARY CROSS ENTROPHY 
              metrics=['accuracy'])  # Accuracy metric

# Model Summary
model.summary()

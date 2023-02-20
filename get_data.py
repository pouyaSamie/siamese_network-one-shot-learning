import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def getData(data_dir, target_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # Splitting the data into training and validation
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=32,
        class_mode='categorical',
        subset='training'  # Use the training subset of the data
    )

    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=32,
        class_mode='categorical',
        subset='validation'  # Use the validation subset of the data
    )

    return train_generator, val_generator

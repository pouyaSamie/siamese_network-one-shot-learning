import tensorflow as tf
from Siamese_model import get_siamese_model

def train_model(train_generator, val_generator, input_shape, model_weights_path, epochs=10):
    """
    Trains a Siamese model on the given data generators.

    Args:
    - train_generator: The data generator for training data.
    - val_generator: The data generator for validation data.
    - input_shape (tuple): The shape of the input images.
    - model_weights_path (str): The file path to save the trained model weights.
    - epochs (int): The number of epochs to train the model for. Default is 10.
    """

    # Get the Siamese model
    siamese_model = get_siamese_model(input_shape)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(lr=0.00006)
    siamese_model.compile(loss='binary_crossentropy', optimizer=optimizer)

    # Train the model
    history = siamese_model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // val_generator.batch_size,
        epochs=epochs
    )

    # Save the trained model weights
    siamese_model.save_weights(model_weights_path)

    return history

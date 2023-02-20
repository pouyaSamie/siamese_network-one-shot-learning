import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Lambda, Dense, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def get_siamese_model(input_shape):
    """
    Creates a Siamese model for image similarity.

    Args:
    - input_shape (tuple): The shape of the input images.

    Returns:
    - model (keras.Model): A Siamese model for image similarity.
    """
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Define the convolutional neural network layers
    model = tf.keras.Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(2e-4)),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(2e-4)),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(2e-4)),
        MaxPooling2D(),
        Dropout(0.2),
        Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(2e-4)),
        Flatten(),
        Dense(4096, activation='sigmoid', kernel_regularizer=l2(1e-3))
    ])

    # Create the left feature extractor
    encoded_l = model(left_input)

    # Create the right feature extractor
    encoded_r = model(right_input)

    # Use the L1 distance function to combine the two encoded features
    L1_layer = Lambda(lambda tensor: tf.abs(tensor[0] - tensor[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Create the final classification layer
    prediction = Dense(1, activation='sigmoid')(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    return siamese_net

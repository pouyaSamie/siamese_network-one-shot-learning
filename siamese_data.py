import numpy as np
import tensorflow as tf


def get_label(dataset, class_name):
        return dataset.class_indices[class_name]



def prepare_siamese_data(generator, n_way, n_shot, n_query, input_shape):
    # if split == 'train':
    #     generator = train_generator
    # else:
    #     generator = val_generator

    classes = list(generator.class_indices.keys())
    num_classes = len(classes)

    # Create empty arrays for query and support images and their corresponding labels
    query_images = np.zeros((n_query*num_classes, *input_shape))
    support_images = np.zeros((n_way, n_shot, *input_shape))
    query_labels = np.zeros(n_query*num_classes)
    support_labels = np.zeros((n_way, n_shot))

    for i in range(num_classes):
        # Get the class name
        class_name = classes[i]

        # Get the label for this class
        class_label = get_label(generator, class_name)

        # Get n_shot images of this class from the generator
        for j in range(n_way):
            class_images = generator.next()[0]
            support_images[j] = class_images[:n_shot]

            # Set the label of these support images
            support_labels[j] = class_label

        # Get n_query images of this class from the generator
        for j in range(n_query):
            class_images = generator.next()[0]
            query_images[i*n_query + j] = class_images[0]

            # Set the label of these query images
            query_labels[i*n_query + j] = class_label

    # Normalize the images
    query_images /= 255.0
    support_images /= 255.0

    # Convert the labels to one-hot encoding
    query_labels = tf.keras.utils.to_categorical(query_labels, num_classes)
    support_labels = tf.keras.utils.to_categorical(support_labels, num_classes)

    # Create the training or validation dataset
    if generator.subset == 'training':
        siamese_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_1': support_images.reshape((n_way*n_shot, *input_shape)),
                'input_2': query_images
            },
            support_labels.reshape(n_way*n_shot, num_classes)
        ))
        siamese_dataset = siamese_dataset.shuffle(100).batch(32)
    else:
        siamese_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_1': support_images.reshape((n_way*n_shot, *input_shape)),
                'input_2': query_images
            },
            query_labels
        ))
        siamese_dataset = siamese_dataset.batch(32)

    return siamese_dataset

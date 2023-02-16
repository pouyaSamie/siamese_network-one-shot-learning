import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image

def load_images(folder_path):
    images = []
    labels = []
    label_map = {}
    for i, folder_name in enumerate(os.listdir(folder_path)):
        folder_path_ = os.path.join(folder_path, folder_name)
        if not os.path.isdir(folder_path_):
            continue
        label_map[i] = folder_name
        for image_name in os.listdir(folder_path_):
            image_path = os.path.join(folder_path_, image_name)
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
            images.append(image)
            labels.append(i)
    return np.array(images), np.array(labels), label_map

def create_siamese_network(input_shape):
    input_1 = keras.Input(shape=input_shape)
    input_2 = keras.Input(shape=input_shape)

    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    flatten = layers.Flatten()
    output = layers.Dense(1, activation="sigmoid")

    x1 = base_model(input_1)
    x1 = flatten(x1)
    x1 = output(x1)

    x2 = base_model(input_2)
    x2 = flatten(x2)
    x2 = output(x2)

    distance = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([x1, x2])
    model = keras.Model(inputs=[input_1, input_2], outputs=distance)
    return model

def generate_pairs(data_dir):
    
    subfolders = [os.path.abspath(os.path.join(data_dir, d)) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    pairs = []
    labels = []
    for subfolder in subfolders:
        images = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.endswith('.jpg') or f.endswith('.png')]

        # Generate pairs of images from this subfolder
        for i, img1_path in enumerate(images):
            for j, img2_path in enumerate(images[i+1:]):
                pairs.append((img1_path, img2_path))
                labels.append(1 if os.path.basename(img1_path).split('_')[0] == os.path.basename(img2_path).split('_')[0] else 0)

        # Generate additional pairs based on random selections from available images
        while len(images) > 1:
            img1_path = images.pop(0)
            for img2_path in images:
                pairs.append((img1_path, img2_path))
                labels.append(1 if os.path.basename(img1_path).split('_')[0] == os.path.basename(img2_path).split('_')[0] else 0)

            # Generate one additional pair with random image selection
            img2_path = random.choice(images)
            pairs.append((img1_path, img2_path))
            labels.append(0)

    return pairs, labels

def train_siamese_network(images, model):
    pairs, y = generate_pairs(images)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit([pairs[:, 0], pairs[:, 1]], y, epochs=10, batch_size=16)


def test_siamese_network(test_image_path, reference_images, model, label_map):
    test_image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=input_shape[:2])
    test_image = tf.keras.preprocessing.image.img_to_array(test_image)

    # Compute feature embeddings for the test image
    test_image_embedding = model.predict(np.expand_dims(test_image, axis=0))

    # Compute feature embeddings for the reference images
    reference_embeddings = model.predict(reference_images)

    # Compute similarity scores between the test image and the reference images
    similarity_scores = cosine_similarity(test_image_embedding, reference_embeddings)

    # Find the index of the most similar image
    max_index = np.argmax(similarity_scores)

    # Map the index to the corresponding class label
    predicted_label = label_map[max_index]

    return predicted_label



# model = create_siamese_network(input_shape)
# train_siamese_network(images, labels, model)

# test_image_path = "test_image.jpg"
# test_image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(224, 224))
# test_image = tf.keras.preprocessing.image.img_to_array(test_image)
# test_image = tf.keras.applications.mobilenet_v2.preprocess_input(test_image)

# reference_images = []
# for i in range(len(label_map)):
#     class_name = label_map[i]
#     class_path = os.path.join("images", class_name)
#     reference_image_path = os.path.join(class_path, os.listdir(class_path)[0])
#     reference_image = tf.keras.preprocessing.image.load_img(reference_image_path, target_size=(224, 224))
#     reference_image = tf.keras.preprocessing.image.img_to_array(reference_image)
#     reference_image = tf.keras.applications.mobilenet_v2.preprocess_input(reference_image)
#     reference_images.append(reference_image)

# predicted_class = test_siamese_network(test_image, reference_images, model, label_map)
# print(predicted_class)
directory = r'images'
images, labels, label_map = load_images(directory)
input_shape = images.shape[1:]

model = create_siamese_network(input_shape)
train_siamese_network(images,model)

# Change the test_image
test_image_path = os.path.join(directory, 'Abomasnow', '3.jpg')
test_image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(224, 224))
test_image = tf.keras.preprocessing.image.img_to_array(test_image)

reference_images = images[labels == np.argmax(labels)]
class_name = test_siamese_network(test_image, reference_images, model, label_map)
print('The predicted class is', class_name)

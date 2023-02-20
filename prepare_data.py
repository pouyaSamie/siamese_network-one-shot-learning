import os
import random
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def prepare_data(input_dir, output_dir, target_size=(224, 224)):
    """
    Resizes images to the specified target size and saves them in a new output directory while maintaining the subfolder structure.

    Args:
    - input_dir (str): The input directory containing the subfolders of images to resize.
    - output_dir (str): The output directory to save the resized images in.
    - target_size (tuple of ints): The target size to resize the images to. Default is (224, 224).
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for sub_dir in dirs:
            new_root = root.replace(input_dir, output_dir)
            new_dir = os.path.join(new_root, sub_dir)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

        for file in files:
            if not file.endswith('.jpg'):
                continue
            new_root = root.replace(input_dir, output_dir)
            new_file = os.path.join(new_root, file)
            if os.path.exists(new_file):
                continue
            try:
                with Image.open(os.path.join(root, file)) as im:
                    im = im.resize(target_size, Image.ANTIALIAS)
                    im.save(new_file, "JPEG")
            except Exception as e:
                print(f"Error processing {os.path.join(root, file)}: {e}")

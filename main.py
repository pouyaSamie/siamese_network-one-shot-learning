from get_data import getData
from prepare_data import prepare_data
from siamese_data import prepare_siamese_data
from Siamese_model import get_siamese_model

import os
os.environ["KMP_DUPLICATE_LIB_OK"] ="TRUE"
# Set paths to data and output directories
data_dir = 'images'
output_dir = 'Data'
target_size = (224, 224)
n_way = 5
n_shot = 5
n_query = 5
input_shape = (224, 224, 3)

# Prepare the data
train_generator, val_generator = getData(output_dir,target_size)

# Create training and validation datasets for Siamese network
train_siamese_dataset = prepare_siamese_data(train_generator, n_way, n_shot, n_query, input_shape)
val_siamese_dataset = prepare_siamese_data(val_generator, n_way, n_shot, n_query, input_shape)

# Get the Siamese model
siamese_model = get_siamese_model(input_shape)

# Train the Siamese network
siamese_model.fit(
    train_siamese_dataset,
    validation_data=val_siamese_dataset,
    epochs=10,
)

# Predict the classes of query images using the trained Siamese model
predictions = []
for x, y in val_siamese_dataset:
    pred = siamese_model.predict([x[:, 0], x[:, 1]])
    predictions.append(pred)
    
print(predictions)

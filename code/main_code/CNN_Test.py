import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error

# Set the paths
model_path = '/root/.ipython/WegnerThesis/data/CNN_Data/cnn_model.h5'
data_dir = "/root/.ipython/WegnerThesis/data/unzipped_data/CUB_200_2011/CUB_200_2011"
images_dir = os.path.join(data_dir, 'images')
labels_file = os.path.join(data_dir, 'image_class_labels.txt')
images_file = os.path.join(data_dir, 'images.txt')
attributes_file = os.path.join(data_dir, 'attributes/class_attribute_labels_continuous.txt')
train_test_split_file = os.path.join(data_dir, 'train_test_split.txt')
class_names_file = os.path.join(data_dir, 'classes.txt')
output_csv_path = "/root/.ipython/WegnerThesis/charts_figures_etc/predicted_vs_actual_test_images.csv"

# Load the model
model = tf.keras.models.load_model(model_path)

# Image settings
img_height, img_width = 224, 224

# Load mappings
image_id_to_class = pd.read_csv(labels_file, delim_whitespace=True, header=None, names=['image_id', 'class_id'])
image_id_to_filename = pd.read_csv(images_file, delim_whitespace=True, header=None, names=['image_id', 'filename'])
class_id_to_attributes = pd.read_csv(attributes_file, delim_whitespace=True, header=None)
class_id_to_name = pd.read_csv(class_names_file, delim_whitespace=True, header=None, names=['class_id', 'class_name'])

# Load train/test split
train_test_split = pd.read_csv(train_test_split_file, delim_whitespace=True, header=None, names=['image_id', 'is_training'])

# Merge to get a complete dataset
dataset = image_id_to_class.merge(image_id_to_filename, on='image_id').merge(train_test_split, on='image_id')
dataset = dataset.merge(class_id_to_attributes, left_on='class_id', right_index=True)
dataset = dataset.merge(class_id_to_name, on='class_id')

# Filter to get only test data
test_data = dataset[dataset['is_training'] == 0]

def preprocess_image(img_path):
    """ Load and preprocess image """
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

def load_images_and_labels(data):
    """ Load images and corresponding labels """
    images = []
    labels = []
    bird_names = []

    for _, row in data.iterrows():
        img_path = os.path.join(images_dir, row['filename'])
        img = preprocess_image(img_path)
        images.append(img)
        labels.append(row.iloc[4:-1].values.astype(float) / 100.0)  # Convert percentage to [0, 1] range
        bird_names.append(row['class_name'])

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32), bird_names

# Load test data
test_images, test_labels, test_bird_names = load_images_and_labels(test_data)

# Randomly select 20 test images
random_indices = np.random.choice(len(test_images), 20, replace=False)
selected_test_images = test_images[random_indices]
selected_test_labels = test_labels[random_indices]
selected_bird_names = [test_bird_names[i] for i in random_indices]

# Predict using the model
predictions = model.predict(selected_test_images)
predicted_percentages = predictions * 100  # Convert back to percentage

# Convert the true labels back to percentage
true_percentages = selected_test_labels * 100

# Calculate MSE between predicted and true labels
mse = mean_squared_error(true_percentages, predicted_percentages)
print(f"Mean Squared Error on Selected Test Images: {mse:.4f}")

# Create a DataFrame to store the true and predicted labels
output_data = []

for i in range(len(selected_test_images)):
    output_data.append([selected_bird_names[i], "true"] + list(true_percentages[i]))
    output_data.append([selected_bird_names[i], "predicted"] + list(predicted_percentages[i]))

# Save the DataFrame as a CSV file
output_df = pd.DataFrame(output_data)
output_df.to_csv(output_csv_path, index=False, header=False)

print(f"Predicted vs Actual values saved to {output_csv_path}")

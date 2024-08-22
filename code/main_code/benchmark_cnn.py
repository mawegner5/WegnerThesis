import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tensorflow.keras import layers, models, optimizers, callbacks
import matplotlib.pyplot as plt

# Set paths
data_dir = "/root/.ipython/WegnerThesis/data/unzipped_data/CUB_200_2011/CUB_200_2011"
images_dir = os.path.join(data_dir, 'images')
labels_file = os.path.join(data_dir, 'image_class_labels.txt')
images_file = os.path.join(data_dir, 'images.txt')
attributes_file = os.path.join(data_dir, 'attributes/class_attribute_labels_continuous.txt')
train_test_split_file = os.path.join(data_dir, 'train_test_split.txt')
class_names_file = os.path.join(data_dir, 'classes.txt')

output_dir = "/root/.ipython/WegnerThesis/charts_figures_etc"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_save_path = os.path.join(output_dir, 'benchmark_ssl4eo_s12_model.h5')
output_csv_path = os.path.join(output_dir, 'benchmark_ssl4eo_s12_predictions.csv')

# Image settings
img_height, img_width = 224, 224
batch_size = 32
epochs = 50
learning_rate = 0.001

# Load data
image_id_to_class = pd.read_csv(labels_file, delim_whitespace=True, header=None, names=['image_id', 'class_id'])
image_id_to_filename = pd.read_csv(images_file, delim_whitespace=True, header=None, names=['image_id', 'filename'])
class_id_to_attributes = pd.read_csv(attributes_file, delim_whitespace=True, header=None)
class_id_to_name = pd.read_csv(class_names_file, delim_whitespace=True, header=None, names=['class_id', 'class_name'])
train_test_split = pd.read_csv(train_test_split_file, delim_whitespace=True, header=None, names=['image_id', 'is_training'])

# Merge to create a complete dataset
dataset = image_id_to_class.merge(image_id_to_filename, on='image_id').merge(train_test_split, on='image_id')
dataset = dataset.merge(class_id_to_attributes, left_on='class_id', right_index=True)
dataset = dataset.merge(class_id_to_name, on='class_id')

# Preprocess images
def preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

# Load images and labels
def load_images_and_labels(data):
    images = []
    labels = []
    bird_names = []

    for _, row in data.iterrows():
        img_path = os.path.join(images_dir, row['filename'])
        img = preprocess_image(img_path)
        images.append(img)
        labels.append(row.iloc[4:-1].values.astype(float) / 100.0)  # Convert to [0, 1] range
        bird_names.append(row['class_name'])

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32), bird_names

# Split data into training and test sets
train_data = dataset[dataset['is_training'] == 1]
test_data = dataset[dataset['is_training'] == 0]

train_images, train_labels, _ = load_images_and_labels(train_data)
test_images, test_labels, test_bird_names = load_images_and_labels(test_data)

# Define the SSL4EO-S12-inspired model
def create_ssl4eo_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(312, activation='sigmoid'))  # Output layer for attributes
    return model

# Compile the model
model = create_ssl4eo_model()
model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), 
              loss='mean_squared_error', 
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

# Summary of the model
model.summary()

# Callbacks
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)

# Train the model
history = model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, 
                    validation_data=(test_images, test_labels), callbacks=[lr_scheduler])

# Save the model
model.save(model_save_path)

# Plot training & validation loss
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.savefig(os.path.join(output_dir, 'benchmark_ssl4eo_s12_loss_plot.png'))
plt.close()

# Evaluate the model on test data
test_predictions = model.predict(test_images)
predicted_percentages = test_predictions * 100  # Convert back to percentage

# Calculate MSE between predicted and true labels
mse = mean_squared_error(test_labels * 100, predicted_percentages)
print(f"Mean Squared Error on Test Set: {mse:.4f}")

# Save the predictions and true labels to CSV
output_data = []
for i in range(len(test_images)):
    output_data.append([test_bird_names[i], "true"] + list(test_labels[i] * 100))
    output_data.append([test_bird_names[i], "predicted"] + list(predicted_percentages[i]))

output_df = pd.DataFrame(output_data)
output_df.to_csv(output_csv_path, index=False, header=False)

print(f"Predicted vs Actual values saved to {output_csv_path}")

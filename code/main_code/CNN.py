import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt

# Hyperparameters
epochs = 50
batch_size = 32
learning_rate = 0.001
lr_reduction_factor = 0.1
lr_patience = 5
lr_min = 1e-6
img_height, img_width = 224, 224
dropout_rate = 0.5
conv1_filters = 32
conv2_filters = 64
conv3_filters = 128
dense_units = 256

# Set the paths for the datasets
data_dir = "/root/.ipython/WegnerThesis/data/unzipped_data/CUB_200_2011/CUB_200_2011"
images_dir = os.path.join(data_dir, 'images')
labels_file = os.path.join(data_dir, 'image_class_labels.txt')
images_file = os.path.join(data_dir, 'images.txt')
attributes_file = os.path.join(data_dir, 'attributes/class_attribute_labels_continuous.txt')
train_test_split_file = os.path.join(data_dir, 'train_test_split.txt')

# Load mappings
image_id_to_class = pd.read_csv(labels_file, delim_whitespace=True, header=None, names=['image_id', 'class_id'])
image_id_to_filename = pd.read_csv(images_file, delim_whitespace=True, header=None, names=['image_id', 'filename'])
class_id_to_attributes = pd.read_csv(attributes_file, delim_whitespace=True, header=None)

# Load train/test split
train_test_split = pd.read_csv(train_test_split_file, delim_whitespace=True, header=None, names=['image_id', 'is_training'])

# Merge to get a complete dataset
dataset = image_id_to_class.merge(image_id_to_filename, on='image_id').merge(train_test_split, on='image_id')
dataset = dataset.merge(class_id_to_attributes, left_on='class_id', right_index=True)

# Split into training and validation sets
train_data = dataset[dataset['is_training'] == 1]
valid_data = dataset[dataset['is_training'] == 0]

def preprocess_image(img_path):
    """ Load and preprocess image """
    img = load_img(img_path, target_size=(img_height, img_width))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

def load_images_and_labels(data):
    """ Load images and corresponding labels """
    images = []
    labels = []

    for _, row in data.iterrows():
        img_path = os.path.join(images_dir, row['filename'])
        img = preprocess_image(img_path)
        images.append(img)
        labels.append(row.iloc[4:].values.astype(float) / 100.0)  # Convert percentage to [0, 1] range

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)

# Load training and validation data
train_images, train_labels = load_images_and_labels(train_data)
valid_images, valid_labels = load_images_and_labels(valid_data)

# Debugging information
print(f"Train images shape: {train_images.shape}, Train labels shape: {train_labels.shape}")
print(f"Validation images shape: {valid_images.shape}, Validation labels shape: {valid_labels.shape}")

# Define the CNN model
model = models.Sequential()

# Convolutional layers with MaxPooling
model.add(layers.Conv2D(conv1_filters, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(conv2_filters, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(conv3_filters, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flattening the output and adding Dense layers
model.add(layers.Flatten())
model.add(layers.Dense(dense_units, activation='relu'))
model.add(layers.Dropout(dropout_rate))

# Output layer with 312 neurons (one for each attribute)
model.add(layers.Dense(312, activation='sigmoid'))

# Compile the model with MSE as the loss function
model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
              loss='mean_squared_error',
              metrics=[tf.keras.metrics.RootMeanSquaredError(), 'mean_absolute_percentage_error'])

# Summary of the model
model.summary()

# Learning rate scheduler callback
lr_scheduler = callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=lr_reduction_factor, 
    patience=lr_patience, 
    min_lr=lr_min, 
    verbose=1
)

# Data augmentation for the training set
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the model
history = model.fit(
    train_datagen.flow(train_images, train_labels, batch_size=batch_size),
    steps_per_epoch=len(train_images) // batch_size,
    epochs=epochs,
    validation_data=(valid_images, valid_labels),
    validation_steps=len(valid_images) // batch_size,
    callbacks=[lr_scheduler]
)

# Save the model
model.save('/root/.ipython/WegnerThesis/data/CNN_Data/cnn_model.h5')

# Evaluate the model on the validation data
val_loss, val_rmse, val_mape = model.evaluate(valid_images, valid_labels)

print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Validation MAPE: {val_mape:.4f}")

# Set up the directory to save the plot
plot_dir = "/root/.ipython/WegnerThesis/charts_figures_etc"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Plotting training & validation loss values
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')

# Save the plot as a figure
plt.savefig(os.path.join(plot_dir, 'model_loss_plot.png'))
plt.close()

# Example prediction on validation data
predictions = model.predict(valid_images)
predicted_percentages = predictions * 100  # Convert back to percentage

# Load true labels for comparison
true_labels = valid_labels * 100  # Assuming labels are also normalized

# Calculate MAPE and R² for more context
mape = mean_absolute_percentage_error(true_labels, predicted_percentages)
r2 = r2_score(true_labels, predicted_percentages)

# Print and save the metrics
print(f"Mean Absolute Percentage Error (MAPE) on Validation Set: {mape:.4f}")
print(f"Coefficient of Determination (R²) on Validation Set: {r2:.4f}")

with open(os.path.join(plot_dir, 'validation_metrics.txt'), 'w') as f:
    f.write(f"Mean Absolute Percentage Error (MAPE) on Validation Set: {mape:.4f}\n")
    f.write(f"Coefficient of Determination (R²) on Validation Set: {r2:.4f}\n")

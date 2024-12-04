import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Define file paths
base_dir = "/root/.ipython/WegnerThesis/data/unzipped_data/CUB_200_2011/CUB_200_2011/"
image_dir = os.path.join(base_dir, "images")
images_file = os.path.join(base_dir, "images.txt")
train_test_split_file = os.path.join(base_dir, "train_test_split.txt")
class_labels_file = os.path.join(base_dir, "image_class_labels.txt")
output_dir = "/root/.ipython/WegnerThesis/data/CNN_Data"

# Create output directories for train, validation, and test sets
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "validation")
test_dir = os.path.join(output_dir, "test")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Load image paths and class labels
images_df = pd.read_csv(images_file, delim_whitespace=True, header=None, names=["image_id", "file_name"])
class_labels_df = pd.read_csv(class_labels_file, delim_whitespace=True, header=None, names=["image_id", "class_id"])
train_test_split_df = pd.read_csv(train_test_split_file, delim_whitespace=True, header=None, names=["image_id", "is_training_image"])

# Merge dataframes to have all relevant information in one dataframer
data_df = pd.merge(images_df, class_labels_df, on="image_id")
data_df = pd.merge(data_df, train_test_split_df, on="image_id")

# Split the training data further into training and validation sets (80% train, 20% validation)
train_df = data_df[data_df['is_training_image'] == 1]
train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['class_id'], random_state=42)

# Test set remains the same
test_df = data_df[data_df['is_training_image'] == 0]

# Save images and labels for train set
for _, row in train_df.iterrows():
    src_path = os.path.join(image_dir, row['file_name'])
    dst_path = os.path.join(train_dir, f"class_{row['class_id']}", os.path.basename(row['file_name']))
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy(src_path, dst_path)

# Save images and labels for validation set
for _, row in val_df.iterrows():
    src_path = os.path.join(image_dir, row['file_name'])
    dst_path = os.path.join(val_dir, f"class_{row['class_id']}", os.path.basename(row['file_name']))
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy(src_path, dst_path)

# Save images and labels for test set
for _, row in test_df.iterrows():
    src_path = os.path.join(image_dir, row['file_name'])
    dst_path = os.path.join(test_dir, f"class_{row['class_id']}", os.path.basename(row['file_name']))
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy(src_path, dst_path)

print(f"Data has been split into train, validation, and test sets and saved to {output_dir}.")

import os
import sys
import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# --------------------------
# User-Modifiable Parameters
# --------------------------

# Data directories
data_dir = '/remote_home/WegnerThesis/animals_with_attributes/animals_w_att_data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validate')
test_dir = os.path.join(data_dir, 'test')

# Output directory for saving predictions and reports
output_dir = '/remote_home/WegnerThesis/charts_figures_etc'

# Path to the attribute CSV file
attributes_csv_path = os.path.join(data_dir, 'predicate_matrix_with_labels.csv')

# Number of images to check per class
num_images_to_check = 5  # Adjust as needed

# Whether to display sample images
display_samples = True  # Set to True to display sample images

# --------------------------
#       End of User Settings
# --------------------------

def check_directories():
    print("Checking directories...")

    # Check data directories
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        sys.exit(1)
    else:
        print(f"Data directory {data_dir} exists.")

    for subdir in ['train', 'validate', 'test']:
        dir_path = os.path.join(data_dir, subdir)
        if not os.path.exists(dir_path):
            print(f"Directory {dir_path} does not exist.")
            sys.exit(1)
        else:
            print(f"Directory {dir_path} exists.")

    # Check output directory
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist. Creating it.")
        os.makedirs(output_dir)
    else:
        print(f"Output directory {output_dir} exists.")

    # Check attribute CSV file
    if not os.path.exists(attributes_csv_path):
        print(f"Attribute CSV file {attributes_csv_path} does not exist.")
        sys.exit(1)
    else:
        print(f"Attribute CSV file {attributes_csv_path} exists.")

def load_attributes():
    print("\nLoading attribute data...")
    attributes_df = pd.read_csv(attributes_csv_path, index_col=0)
    print(f"Attributes DataFrame loaded with shape {attributes_df.shape}.")

    # Adjust class names to match directory names (replace spaces with '+')
    attributes_df.index = attributes_df.index.str.replace(' ', '+')
    print("Adjusted class names in attributes DataFrame to match directory names.")

    attribute_names = attributes_df.columns.tolist()
    classes = attributes_df.index.tolist()
    num_attributes = len(attribute_names)
    num_classes = len(classes)

    print(f"Number of attributes: {num_attributes}")
    print(f"Number of classes: {num_classes}")

    return attributes_df, attribute_names, classes

def check_class_directories(classes, phase_dir):
    print(f"\nChecking class directories in {phase_dir}...")

    dir_classes = os.listdir(phase_dir)
    dir_classes = [d for d in dir_classes if os.path.isdir(os.path.join(phase_dir, d))]
    print(f"Found {len(dir_classes)} class directories in {phase_dir}.")

    # Classes in the directory that are not in the attribute list
    unexpected_classes = set(dir_classes) - set(classes)
    if unexpected_classes:
        print(f"Warning: Found unexpected classes in {phase_dir}: {unexpected_classes}")

    # Classes in the attribute list that are not in the directory
    missing_classes = set(classes) - set(dir_classes)
    if missing_classes:
        print(f"Warning: Missing classes in {phase_dir}: {missing_classes}")

    return dir_classes

def check_images_in_class(class_dir, class_name, attributes_df):
    images = os.listdir(class_dir)
    images = [img for img in images if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not images:
        print(f"No image files found in {class_dir}.")
        return

    # Check that the attributes exist for this class
    if class_name not in attributes_df.index:
        print(f"Attributes not found for class {class_name}.")
        return

    # Load a few images to check
    for img_name in images[:num_images_to_check]:
        img_path = os.path.join(class_dir, img_name)
        try:
            img = Image.open(img_path)
            img.verify()  # Verify that it's a valid image
            img = Image.open(img_path)  # Reopen for operations after verify()
            img = img.convert('RGB')  # Ensure it's in RGB format
            print(f"Loaded image {img_name} from {class_name} successfully.")
        except Exception as e:
            print(f"Error loading image {img_name} from {class_name}: {e}")

def check_images(classes, attributes_df, phase):
    phase_dir = os.path.join(data_dir, phase)
    dir_classes = check_class_directories(classes, phase_dir)

    for class_name in dir_classes:
        class_dir = os.path.join(phase_dir, class_name)
        check_images_in_class(class_dir, class_name, attributes_df)

def display_sample_images(classes, attributes_df, phase):
    if not display_samples:
        return

    print(f"\nDisplaying sample images from {phase} set...")
    phase_dir = os.path.join(data_dir, phase)
    dir_classes = os.listdir(phase_dir)
    dir_classes = [d for d in dir_classes if os.path.isdir(os.path.join(phase_dir, d))]

    for class_name in dir_classes[:3]:  # Display samples from first 3 classes
        class_dir = os.path.join(phase_dir, class_name)
        images = os.listdir(class_dir)
        images = [img for img in images if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            continue

        img_name = images[0]
        img_path = os.path.join(class_dir, img_name)
        img = Image.open(img_path).convert('RGB')

        attributes = attributes_df.loc[class_name]
        attribute_list = ', '.join([attr for attr, val in attributes.items() if val == 1])

        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.title(f"Class: {class_name}\nAttributes: {attribute_list}")
        plt.axis('off')
        plt.show()

def main():
    check_directories()
    attributes_df, attribute_names, classes = load_attributes()

    print("\nChecking training data...")
    check_images(classes, attributes_df, 'train')

    print("\nChecking validation data...")
    check_images(classes, attributes_df, 'validate')

    print("\nChecking test data...")
    check_images(classes, attributes_df, 'test')

    display_sample_images(classes, attributes_df, 'train')

    print("\nAll data checks completed successfully.")

if __name__ == '__main__':
    main()

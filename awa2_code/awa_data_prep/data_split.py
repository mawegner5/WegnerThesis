import os
import shutil
import random

# Define paths
base_dir = "/remote_home/WegnerThesis/animals_with_attributes/Animals_with_Attributes2"
jpeg_images_dir = os.path.join(base_dir, "JPEGImages")
test_dir = os.path.join(base_dir, "test")
train_dir = os.path.join(base_dir, "train")
validate_dir = os.path.join(base_dir, "validate")

# Class files
test_classes_file = os.path.join(base_dir, "testclasses.txt")
train_classes_file = os.path.join(base_dir, "trainclasses.txt")

# Ensure destination directories exist
os.makedirs(test_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validate_dir, exist_ok=True)

def load_classes(file_path):
    """Load class names from a file."""
    with open(file_path, "r") as file:
        return [line.strip() for line in file]

def move_class_folders(class_names, source_dir, destination_dir):
    """Move folders corresponding to class names."""
    for class_name in class_names:
        src_path = os.path.join(source_dir, class_name)
        dst_path = os.path.join(destination_dir, class_name)
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
        else:
            print(f"Warning: {src_path} does not exist.")

def main():
    # Load class lists
    test_classes = load_classes(test_classes_file)
    train_classes = load_classes(train_classes_file)

    # Randomly select 3 classes for validation
    validate_classes = random.sample(train_classes, 3)

    # Remaining classes will be used for training
    train_classes = [cls for cls in train_classes if cls not in validate_classes]

    # Move test classes
    print("Moving test classes...")
    move_class_folders(test_classes, jpeg_images_dir, test_dir)

    # Move validation classes
    print("Moving validation classes...")
    move_class_folders(validate_classes, jpeg_images_dir, validate_dir)

    # Move training classes
    print("Moving training classes...")
    move_class_folders(train_classes, jpeg_images_dir, train_dir)

    print("Data split complete.")

if __name__ == "__main__":
    main()

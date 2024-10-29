import os
import sys
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.append('/root/.ipython/WegnerThesis/code/main_code/')
from awa2dataset import AWA2Dataset  # Ensure this is the corrected version
import psutil  # For checking system memory
import shutil

def check_directories(root_dir):
    print("\n--- Checking Directories ---")
    if not os.path.exists(root_dir):
        print(f"[Error] Root directory '{root_dir}' does not exist.")
        return False
    else:
        print(f"[OK] Root directory '{root_dir}' exists.")

    # Paths to check
    classes_path = os.path.join(root_dir, '../Animals_with_Attributes2/classes.txt')
    attributes_path = os.path.join(root_dir, '../Animals_with_Attributes2/predicate-matrix-binary.txt')
    images_dir = os.path.join(root_dir, '../Animals_with_Attributes2/JPEGImages')

    # Check classes.txt
    if not os.path.exists(classes_path):
        print(f"[Error] classes.txt not found at '{classes_path}'.")
        return False
    else:
        print(f"[OK] Found classes.txt at '{classes_path}'.")

    # Check predicate-matrix-binary.txt
    if not os.path.exists(attributes_path):
        print(f"[Error] predicate-matrix-binary.txt not found at '{attributes_path}'.")
        return False
    else:
        print(f"[OK] Found predicate-matrix-binary.txt at '{attributes_path}'.")

    # Check JPEGImages directory
    if not os.path.exists(images_dir):
        print(f"[Error] JPEGImages directory not found at '{images_dir}'.")
        return False
    else:
        print(f"[OK] Found JPEGImages directory at '{images_dir}'.")

    return True

def check_class_directories(root_dir):
    print("\n--- Checking Class Directories ---")
    classes_path = os.path.join(root_dir, '../Animals_with_Attributes2/classes.txt')
    images_dir = os.path.join(root_dir, '../Animals_with_Attributes2/JPEGImages')

    # Load class names
    classes = []
    with open(classes_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                cls_name = parts[1]
                classes.append(cls_name)
            else:
                print(f"[Warning] Malformed line in classes.txt: {line.strip()}")

    missing_dirs = []
    for cls_name in classes:
        cls_dir = os.path.join(images_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"[Warning] Class directory '{cls_dir}' not found.")
            missing_dirs.append(cls_name)
        else:
            print(f"[OK] Found class directory '{cls_dir}'.")

    if missing_dirs:
        print(f"\n[Error] Missing class directories for classes: {missing_dirs}")
        return False
    return True

def check_images(root_dir):
    print("\n--- Checking Images in Class Directories ---")
    images_dir = os.path.join(root_dir, '../Animals_with_Attributes2/JPEGImages')
    total_images = 0
    broken_images = []
    for cls_name in os.listdir(images_dir):
        cls_dir = os.path.join(images_dir, cls_name)
        if os.path.isdir(cls_dir):
            image_files = os.listdir(cls_dir)
            if not image_files:
                print(f"[Warning] No images found in '{cls_dir}'.")
            else:
                for img_name in image_files:
                    img_path = os.path.join(cls_dir, img_name)
                    try:
                        with Image.open(img_path) as img:
                            img.verify()  # Verify that it's an image
                        total_images += 1
                    except Exception as e:
                        print(f"[Error] Cannot open image '{img_path}': {e}")
                        broken_images.append(img_path)
    print(f"\n[OK] Total images found: {total_images}")
    if broken_images:
        print(f"[Error] Broken images detected: {broken_images}")
        return False
    return True

def check_attributes(root_dir):
    print("\n--- Checking Attributes ---")
    attributes_path = os.path.join(root_dir, '../Animals_with_Attributes2/predicate-matrix-binary.txt')
    classes_path = os.path.join(root_dir, '../Animals_with_Attributes2/classes.txt')

    # Load attributes matrix
    attribute_matrix = np.loadtxt(attributes_path, dtype=int)
    print(f"[OK] Attribute matrix shape: {attribute_matrix.shape}")

    # Load class names
    classes = []
    with open(classes_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                classes.append(parts[1])

    if attribute_matrix.shape[0] != len(classes):
        print(f"[Error] Number of classes ({len(classes)}) does not match number of rows in attribute matrix ({attribute_matrix.shape[0]}).")
        return False
    else:
        print("[OK] Number of classes matches number of rows in attribute matrix.")

    return True

def test_dataset(root_dir):
    print("\n--- Testing Dataset Initialization ---")
    # Update root_dir to point to the correct location
    dataset = AWA2Dataset(root_dir=os.path.join(root_dir, '../Animals_with_Attributes2/JPEGImages'), transform=None)
    if len(dataset) == 0:
        print("[Error] Dataset is empty.")
        return False
    else:
        print(f"[OK] Dataset contains {len(dataset)} samples.")
    # Try to get a sample
    try:
        image, label = dataset[0]
        print(f"[OK] Loaded sample image of size {image.size} with label shape {label.shape}.")
    except Exception as e:
        print(f"[Error] Failed to load sample from dataset: {e}")
        return False
    return True

def test_dataloader(root_dir):
    print("\n--- Testing DataLoader ---")
    dataset = AWA2Dataset(root_dir=os.path.join(root_dir, '../Animals_with_Attributes2/JPEGImages'), transform=None)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    try:
        for images, labels in dataloader:
            print(f"[OK] DataLoader is working. Batch size: {images.size(0)}")
            break  # Only test the first batch
    except Exception as e:
        print(f"[Error] DataLoader failed: {e}")
        return False
    return True

def check_system_resources():
    print("\n--- Checking System Resources ---")
    # Check shared memory size
    shm_size = None
    try:
        shm_stats = psutil.virtual_memory()
        shm_size = getattr(shm_stats, 'shared', None)
        if shm_size:
            print(f"[OK] Shared memory size: {shm_size / (1024 ** 3):.2f} GB")
        else:
            print("[Warning] Could not determine shared memory size.")
    except Exception as e:
        print(f"[Warning] Failed to get shared memory size: {e}")

    # Check total memory
    total_mem = psutil.virtual_memory().total
    print(f"[OK] Total system memory: {total_mem / (1024 ** 3):.2f} GB")
    return True

def main():
    root_dir = '/root/.ipython/WegnerThesis/animals_with_attributes/animals_w_att_data/train'
    all_checks_passed = True

    # Check directories
    if not check_directories(root_dir):
        all_checks_passed = False

    # Check class directories
    if not check_class_directories(root_dir):
        all_checks_passed = False

    # Check images
    if not check_images(root_dir):
        all_checks_passed = False

    # Check attributes
    if not check_attributes(root_dir):
        all_checks_passed = False

    # Test dataset
    if not test_dataset(root_dir):
        all_checks_passed = False

    # Test DataLoader
    if not test_dataloader(root_dir):
        all_checks_passed = False

    # Check system resources
    check_system_resources()

    if all_checks_passed:
        print("\nAll pre-requisite checks passed. You can proceed to train your CNN.")
    else:
        print("\nSome checks failed. Please address the above issues before proceeding.")

if __name__ == '__main__':
    main()

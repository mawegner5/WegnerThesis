import os
import sys
import importlib
import pkg_resources
from PIL import Image
import numpy as np

def check_packages(required_packages):
    print("\n--- Checking Required Packages ---")
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    missing_packages = []
    for package in required_packages:
        if package.lower() not in installed_packages:
            print(f"[Error] Required package '{package}' is not installed.")
            missing_packages.append(package)
        else:
            print(f"[OK] Package '{package}' is installed.")
    if missing_packages:
        print("\nPlease install the missing packages before proceeding.")
        return False
    return True

def check_files(file_paths):
    print("\n--- Checking Required Files ---")
    all_files_exist = True
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            print(f"[Error] Required file '{file_path}' not found.")
            all_files_exist = False
        else:
            print(f"[OK] Found file '{file_path}'.")
    return all_files_exist

def check_directories(dir_paths):
    print("\n--- Checking Required Directories ---")
    all_dirs_exist = True
    for dir_path in dir_paths:
        if not os.path.isdir(dir_path):
            print(f"[Error] Required directory '{dir_path}' not found.")
            all_dirs_exist = False
        else:
            print(f"[OK] Found directory '{dir_path}'.")
    return all_dirs_exist

def check_class_names(classes_txt_path, images_dir):
    print("\n--- Checking Class Names Match Directories ---")
    # Load class names from classes.txt
    classes_in_file = []
    with open(classes_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                cls_name = parts[1]
                classes_in_file.append(cls_name)
            else:
                print(f"[Warning] Malformed line in classes.txt: {line.strip()}")
    # Get class directories
    classes_in_dir = [name for name in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, name))]
    # Check for mismatches
    missing_dirs = []
    mismatched_classes = []
    for cls_name in classes_in_file:
        if cls_name not in classes_in_dir:
            print(f"[Error] Class '{cls_name}' from classes.txt does not have a corresponding directory in '{images_dir}'.")
            missing_dirs.append(cls_name)
    for cls_name in classes_in_dir:
        if cls_name not in classes_in_file:
            print(f"[Error] Directory '{cls_name}' in '{images_dir}' does not have a corresponding entry in classes.txt.")
            mismatched_classes.append(cls_name)
    if missing_dirs or mismatched_classes:
        return False
    else:
        print("[OK] All class names match between classes.txt and image directories.")
        return True

def check_attribute_matrix(attributes_path, num_classes, num_attributes):
    print("\n--- Checking Attribute Matrix Dimensions ---")
    attribute_matrix = np.loadtxt(attributes_path, dtype=int)
    if attribute_matrix.shape != (num_classes, num_attributes):
        print(f"[Error] Attribute matrix shape {attribute_matrix.shape} does not match expected shape ({num_classes}, {num_attributes}).")
        return False
    else:
        print(f"[OK] Attribute matrix has correct shape ({num_classes}, {num_attributes}).")
        return True

def check_images(images_dir):
    print("\n--- Checking Images in Each Class Directory ---")
    total_images = 0
    corrupted_images = []
    for cls_name in os.listdir(images_dir):
        cls_dir = os.path.join(images_dir, cls_name)
        if os.path.isdir(cls_dir):
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if os.path.isfile(img_path):
                    try:
                        with Image.open(img_path) as img:
                            img.verify()  # Verify that it's an image
                        total_images += 1
                    except Exception as e:
                        print(f"[Error] Cannot open image '{img_path}': {e}")
                        corrupted_images.append(img_path)
                else:
                    print(f"[Warning] '{img_path}' is not a file, skipping...")
    print(f"\n[OK] Total images found: {total_images}")
    if corrupted_images:
        print(f"[Error] Found {len(corrupted_images)} corrupted images.")
        return False
    else:
        print("[OK] All images can be opened successfully.")
        return True

def main():
    # Paths (update these paths if necessary)
    data_dir = '/root/.ipython/WegnerThesis/animals_with_attributes/animals_w_att_data/'
    images_dir = os.path.join(data_dir, 'Animals_with_Attributes2', 'JPEGImages')
    classes_txt_path = os.path.join(data_dir, 'Animals_with_Attributes2', 'classes.txt')
    predicates_txt_path = os.path.join(data_dir, 'Animals_with_Attributes2', 'predicates.txt')
    attributes_path = os.path.join(data_dir, 'Animals_with_Attributes2', 'predicate-matrix-binary.txt')
    output_dir = '/root/.ipython/WegnerThesis/charts_figures_etc/'
    required_directories = [data_dir, images_dir, output_dir]
    required_files = [classes_txt_path, predicates_txt_path, attributes_path]

    # Expected numbers
    expected_num_classes = 50  # Number of classes in AWA2
    expected_num_attributes = 85  # Number of attributes/predicates in AWA2

    # Required packages
    required_packages = ['torch', 'torchvision', 'numpy', 'Pillow', 'matplotlib', 'scikit-learn']

    all_checks_passed = True

    # Check required packages
    if not check_packages(required_packages):
        all_checks_passed = False

    # Check required directories
    if not check_directories(required_directories):
        all_checks_passed = False

    # Check required files
    if not check_files(required_files):
        all_checks_passed = False

    # Check class names match directories
    if not check_class_names(classes_txt_path, images_dir):
        all_checks_passed = False

    # Check attribute matrix dimensions
    if not check_attribute_matrix(attributes_path, expected_num_classes, expected_num_attributes):
        all_checks_passed = False

    # Check images
    if not check_images(images_dir):
        all_checks_passed = False

    if all_checks_passed:
        print("\nAll pre-checks passed. You can proceed to run your CNN training script.")
    else:
        print("\nSome pre-checks failed. Please address the issues above before running your CNN training script.")

if __name__ == '__main__':
    main()

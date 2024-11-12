import os
import sys
import platform
import subprocess
import torch
import torchvision
import numpy as np
from PIL import Image

# ----------------------------
# Environment Checks
# ----------------------------

def check_python_version():
    print(f"Python Version: {platform.python_version()}")
    if sys.version_info < (3, 7):
        print("Warning: Python 3.7 or higher is recommended.")
    else:
        print("Python version is suitable.")

def check_packages():
    required_packages = {
        'torch': 'torch',
        'torchvision': 'torchvision',
        'numpy': 'numpy',
        'PIL': 'PIL',
        'ray': 'ray',
        'tqdm': 'tqdm',
        'sklearn': 'sklearn',
    }
    for pkg_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"Package '{pkg_name}' is installed.")
        except ImportError:
            print(f"Error: Package '{pkg_name}' is not installed.")

def check_package_versions():
    import pkg_resources
    packages = ['torch', 'torchvision', 'numpy', 'Pillow', 'ray', 'tqdm', 'scikit-learn', 'pydantic']
    for pkg in packages:
        try:
            version = pkg_resources.get_distribution(pkg).version
            print(f"{pkg} version: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"Package '{pkg}' not found.")

def check_gpu():
    if torch.cuda.is_available():
        print("CUDA is available.")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Training will be performed on CPU.")

# ----------------------------
# Data Checks
# ----------------------------

def check_data_directories():
    base_dir = '/root/.ipython/WegnerThesis/animals_with_attributes/animals_w_att_data'
    required_dirs = ['train', 'validate', 'test', 'Animals_with_Attributes2']
    for dir_name in required_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.isdir(dir_path):
            print(f"Directory '{dir_path}' exists.")
        else:
            print(f"Error: Directory '{dir_path}' does not exist.")

def check_classes_files():
    base_dir = '/root/.ipython/WegnerThesis/animals_with_attributes/animals_w_att_data/Animals_with_Attributes2'
    files = ['classes.txt', 'predicates.txt', 'predicate-matrix-binary.txt', 'trainclasses.txt', 'testclasses.txt']
    for file_name in files:
        file_path = os.path.join(base_dir, file_name)
        if os.path.isfile(file_path):
            print(f"File '{file_path}' exists.")
        else:
            print(f"Error: File '{file_path}' does not exist.")

def check_class_names():
    classes_txt_path = '/root/.ipython/WegnerThesis/animals_with_attributes/animals_w_att_data/Animals_with_Attributes2/classes.txt'
    with open(classes_txt_path, 'r') as f:
        class_lines = [line.strip().split() for line in f.readlines()]
        class_names = [line[1].replace('+', ' ') for line in class_lines if len(line) >= 2]
    directories = os.listdir('/root/.ipython/WegnerThesis/animals_with_attributes/animals_w_att_data/Animals_with_Attributes2/JPEGImages')
    dir_set = set(directories)
    class_set = set(class_names)
    if class_set == dir_set:
        print("Class names and directories match.")
    else:
        missing_in_dirs = class_set - dir_set
        missing_in_classes = dir_set - class_set
        if missing_in_dirs:
            print(f"Directories missing for classes: {missing_in_dirs}")
        if missing_in_classes:
            print(f"Classes missing in classes.txt: {missing_in_classes}")

def verify_image_files():
    base_dirs = ['/root/.ipython/WegnerThesis/animals_with_attributes/animals_w_att_data/train',
                 '/root/.ipython/WegnerThesis/animals_with_attributes/animals_w_att_data/validate',
                 '/root/.ipython/WegnerThesis/animals_with_attributes/animals_w_att_data/test']
    for base_dir in base_dirs:
        print(f"Checking images in '{base_dir}'...")
        class_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        for class_dir in class_dirs:
            image_files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
            for image_file in image_files:
                image_path = os.path.join(class_dir, image_file)
                try:
                    with Image.open(image_path) as img:
                        img.verify()
                except (IOError, SyntaxError) as e:
                    print(f"Corrupted image file: {image_path} - {e}")

def check_permissions():
    paths = [
        '/root/.ipython/WegnerThesis/animals_with_attributes/animals_w_att_data/train',
        '/root/.ipython/WegnerThesis/animals_with_attributes/animals_w_att_data/validate',
        '/root/.ipython/WegnerThesis/animals_with_attributes/animals_w_att_data/test',
        '/root/.ipython/WegnerThesis/charts_figures_etc/',
    ]
    for path in paths:
        if os.access(path, os.R_OK | os.W_OK | os.X_OK):
            print(f"Permissions for '{path}' are sufficient.")
        else:
            print(f"Insufficient permissions for '{path}'. Check read/write/execute permissions.")

def check_ray_version():
    try:
        import ray
        version = ray.__version__
        print(f"Ray version: {version}")
    except ImportError:
        print("Error: Ray is not installed.")

def check_pydantic_version():
    try:
        import pydantic
        version = pydantic.__version__
        print(f"Pydantic version: {version}")
    except ImportError:
        print("Error: Pydantic is not installed.")

# ----------------------------
# Main Function
# ----------------------------

if __name__ == '__main__':
    print("Starting pre-requisite checks...\n")

    print("Environment Checks:")
    check_python_version()
    check_gpu()
    check_packages()
    check_package_versions()
    check_ray_version()
    check_pydantic_version()
    print("\nData Checks:")
    check_data_directories()
    check_classes_files()
    check_class_names()
    verify_image_files()
    check_permissions()

    print("\nPre-requisite checks completed.")

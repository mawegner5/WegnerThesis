#!/usr/bin/env python3
# pre_reqs.py

import os
from pathlib import Path
import sys

def main():
    # ----------------------------
    # Define Paths
    # ----------------------------
    # Base directory containing the dataset
    dataset_dir = '/root/.ipython/WegnerThesis/animals_with_attributes/animals_w_att_data/Animals_with_Attributes2'
    
    # Directories for splits
    splits_base_dir = '/root/.ipython/WegnerThesis/animals_with_attributes/animals_w_att_data'
    train_dir = os.path.join(splits_base_dir, 'train')
    val_dir = os.path.join(splits_base_dir, 'validate')
    test_dir = os.path.join(splits_base_dir, 'test')
    
    # Class lists
    train_classes_txt = os.path.join(dataset_dir, 'trainclasses.txt')
    val_classes_txt = os.path.join(dataset_dir, 'valclasses.txt')
    test_classes_txt = os.path.join(dataset_dir, 'testclasses.txt')
    
    # Expected Classes
    expected_classes = {
        'antelope',
        'grizzly+bear',
        'killer+whale',
        'beaver',
        'dalmatian',
        'persian+cat',
        'horse',
        'german+shepherd',
        'blue+whale',
        'siamese+cat',
        'skunk',
        'mole',
        'tiger',
        'hippopotamus',
        'leopard',
        'moose',
        'spider+monkey',
        'humpback+whale',
        'elephant',
        'gorilla',
        'ox',
        'fox',
        'sheep',
        'seal',
        'chimpanzee',
        'hamster',
        'squirrel',
        'rhinoceros',
        'rabbit',
        'bat',
        'giraffe',
        'wolf',
        'chihuahua',
        'rat',
        'weasel',
        'otter',
        'buffalo',
        'zebra',
        'giant+panda',
        'deer',
        'bobcat',
        'pig',
        'lion',
        'mouse',
        'polar+bear',
        'collie',
        'walrus',
        'raccoon',
        'cow',
        'dolphin'
    }
    
    # ----------------------------
    # Read Class Lists Function
    # ----------------------------
    def read_classes(file_path):
        if not os.path.exists(file_path):
            print(f"ERROR: Class list file does not exist: {file_path}")
            sys.exit(1)
        with open(file_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes
    
    # ----------------------------
    # Read Class Lists
    # ----------------------------
    train_classes = read_classes(train_classes_txt)
    val_classes = read_classes(val_classes_txt)
    test_classes = read_classes(test_classes_txt)
    
    # ----------------------------
    # Initialize Flags
    # ----------------------------
    errors_found = False
    
    # ----------------------------
    # Check Class Lists Integrity
    # ----------------------------
    print("\n--- Verifying Class Lists Integrity ---")
    
    # 1. Check the number of classes in each split
    if len(train_classes) != 35:
        print(f"ERROR: Expected 35 training classes, found {len(train_classes)}")
        errors_found = True
    else:
        print("PASS: Correct number of training classes (35)")
    
    if len(val_classes) != 5:
        print(f"ERROR: Expected 5 validation classes, found {len(val_classes)}")
        errors_found = True
    else:
        print("PASS: Correct number of validation classes (5)")
    
    if len(test_classes) != 10:
        print(f"ERROR: Expected 10 test classes, found {len(test_classes)}")
        errors_found = True
    else:
        print("PASS: Correct number of test classes (10)")
    
    # 2. Check for duplicate classes across splits
    train_set = set(train_classes)
    val_set = set(val_classes)
    test_set = set(test_classes)
    
    overlap_train_val = train_set.intersection(val_set)
    overlap_train_test = train_set.intersection(test_set)
    overlap_val_test = val_set.intersection(test_set)
    
    if overlap_train_val:
        print(f"ERROR: Overlapping classes between train and validate: {overlap_train_val}")
        errors_found = True
    else:
        print("PASS: No overlapping classes between train and validate")
    
    if overlap_train_test:
        print(f"ERROR: Overlapping classes between train and test: {overlap_train_test}")
        errors_found = True
    else:
        print("PASS: No overlapping classes between train and test")
    
    if overlap_val_test:
        print(f"ERROR: Overlapping classes between validate and test: {overlap_val_test}")
        errors_found = True
    else:
        print("PASS: No overlapping classes between validate and test")
    
    # 3. Check that all classes are among the expected classes
    all_split_classes = set(train_classes + val_classes + test_classes)
    unexpected_classes = all_split_classes - expected_classes
    missing_classes = expected_classes - all_split_classes
    
    if unexpected_classes:
        print(f"ERROR: Unexpected classes found: {unexpected_classes}")
        errors_found = True
    else:
        print("PASS: All classes in splits are expected")
    
    if missing_classes:
        print(f"ERROR: Missing classes not present in any split: {missing_classes}")
        errors_found = True
    else:
        print("PASS: No missing classes")
    
    # 4. Check that total classes across splits equal expected number
    if len(all_split_classes) != 50:
        print(f"ERROR: Total unique classes across splits is {len(all_split_classes)}, expected 50")
        errors_found = True
    else:
        print("PASS: Total unique classes across splits is 50")
    
    # ----------------------------
    # Check Directory Structures
    # ----------------------------
    print("\n--- Verifying Directory Structures ---")
    
    # Define a helper function to verify directories
    def verify_split_dir(split_dir, split_classes, split_name):
        nonlocal errors_found
        print(f"\nVerifying {split_name} directory: {split_dir}")
        if not os.path.exists(split_dir):
            print(f"ERROR: {split_name} directory does not exist: {split_dir}")
            errors_found = True
            return
        
        # List actual class directories in the split
        actual_classes = set([entry.name for entry in Path(split_dir).iterdir() if entry.is_dir()])
        
        # Expected classes for this split
        expected_split_classes = set(split_classes)
        
        # Check for missing classes in the directory
        missing_in_dir = expected_split_classes - actual_classes
        if missing_in_dir:
            print(f"ERROR: The following classes are missing in {split_name} directory: {missing_in_dir}")
            errors_found = True
        else:
            print(f"PASS: All classes are present in {split_name} directory")
        
        # Check for extra classes in the directory
        extra_in_dir = actual_classes - expected_split_classes
        if extra_in_dir:
            print(f"ERROR: The following unexpected classes are present in {split_name} directory: {extra_in_dir}")
            errors_found = True
        else:
            print(f"PASS: No unexpected classes in {split_name} directory")
        
        # Check each class directory contains at least one image
        for cls in expected_split_classes:
            cls_dir = Path(split_dir) / cls
            if not cls_dir.exists():
                print(f"ERROR: Class directory does not exist: {cls_dir}")
                errors_found = True
                continue
            image_files = [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']]
            if not image_files:
                print(f"ERROR: No image files found in class directory: {cls_dir}")
                errors_found = True
            else:
                print(f"PASS: {len(image_files)} images found in {cls_dir}")
    
    # Verify 'train' directory
    verify_split_dir(train_dir, train_classes, 'Train')
    
    # Verify 'validate' directory
    verify_split_dir(val_dir, val_classes, 'Validate')
    
    # Verify 'test' directory
    verify_split_dir(test_dir, test_classes, 'Test')
    
    # ----------------------------
    # Final Summary
    # ----------------------------
    print("\n--- Summary ---")
    if errors_found:
        print("One or more issues were found during the verification process. Please address them before proceeding to train the model.")
        sys.exit(1)
    else:
        print("All checks passed successfully. Your dataset is correctly organized and ready for training.")
        sys.exit(0)

if __name__ == "__main__":
    main()

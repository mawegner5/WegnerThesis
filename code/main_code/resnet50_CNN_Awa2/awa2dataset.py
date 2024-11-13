# awa2dataset.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class AWA2Dataset(Dataset):
    def __init__(self, class_dirs, predicates_txt_path, attributes_path, sorted_jpeg_dir, transform=None):
        """
        Args:
            class_dirs (list): List of directories, each corresponding to a class containing images.
            predicates_txt_path (str): Path to predicates.txt file.
            attributes_path (str): Path to predicate-matrix-binary.txt file.
            sorted_jpeg_dir (str): Path to JPEGImages directory with all class names.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.transform = transform

        # Load attribute names
        self.attributes = []
        with open(predicates_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    attr_name = parts[1]
                    self.attributes.append(attr_name)

        # Load attribute matrix
        try:
            self.attribute_matrix = np.loadtxt(attributes_path, dtype=int)  # Shape: (N_classes, N_attributes)
            print(f"Loaded attribute matrix with shape: {self.attribute_matrix.shape}")
        except Exception as e:
            raise ValueError(f"Error loading attribute matrix: {e}")

        # Load sorted class names from JPEGImages to ensure alignment with attribute matrix
        if not os.path.isdir(sorted_jpeg_dir):
            raise ValueError(f"JPEGImages directory does not exist: {sorted_jpeg_dir}")

        self.sorted_class_names = sorted(os.listdir(sorted_jpeg_dir))
        if len(self.sorted_class_names) != self.attribute_matrix.shape[0]:
            raise ValueError(f"Number of classes in JPEGImages ({len(self.sorted_class_names)}) does not match attribute matrix ({self.attribute_matrix.shape[0]}).")

        # Create a mapping from class name to index
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.sorted_class_names)}

        # Collect all image paths and labels
        self.image_paths = []
        self.labels = []
        for class_dir in class_dirs:
            if not os.path.isdir(class_dir):
                print(f"Skipping non-directory: {class_dir}")
                continue
            class_name = os.path.basename(class_dir)
            if class_name not in self.class_to_idx:
                print(f"Class name {class_name} not found in sorted class list. Skipping.")
                continue
            class_idx = self.class_to_idx[class_name]
            label = self.attribute_matrix[class_idx]

            # Iterate over all image files in the class directory
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.image_paths.append(img_path)
                    self.labels.append(label)
                else:
                    print(f"Skipping non-image file: {img_path}")

        print(f"Total images found: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Skip corrupted images by using a blank image
            print(f"Error loading image {img_path}: {e}. Using a blank image instead.")
            image = Image.new('RGB', (224, 224))
        label = self.labels[idx]
        label = torch.from_numpy(label).float()
        if self.transform:
            image = self.transform(image)
        return image, label

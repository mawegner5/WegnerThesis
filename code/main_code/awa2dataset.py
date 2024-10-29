import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class AWA2Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        classes_path = os.path.join(root_dir, '../Animals_with_Attributes2/classes.txt')
        attributes_path = os.path.join(root_dir, '../Animals_with_Attributes2/predicate-matrix-binary.txt')

        # Load class and attribute names
        self.classes = []
        self.class_to_idx = {}
        with open(classes_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    idx = int(parts[0]) - 1  # Adjust index if needed
                    cls_name = parts[1]
                    self.classes.append(cls_name)
                    self.class_to_idx[cls_name] = idx
                else:
                    print(f"[Warning] Malformed line in classes.txt: {line.strip()}")

        self.attribute_matrix = np.loadtxt(attributes_path, dtype=int)  # Shape: (N_classes, 85)

        self.image_paths = []
        self.labels = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                print(f"[Warning] Class directory '{cls_dir}' not found, skipping...")
                continue
            class_idx = self.class_to_idx[cls_name]
            label = self.attribute_matrix[class_idx]  # Binary vector for attributes

            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                self.image_paths.append(img_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[Error] Failed to load image {img_path}: {e}")
            # Handle the error as appropriate, e.g., skip or return a default image
            image = Image.new('RGB', (224, 224))  # Placeholder image
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

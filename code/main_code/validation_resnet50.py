import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
import csv
from tqdm import tqdm

# ----------------------------
# Paths and Hyperparameters
# ----------------------------
# Paths
data_dir = '/root/.ipython/WegnerThesis/animals_with_attributes/animals_w_att_data/'
images_dir = os.path.join(data_dir, 'Animals_with_Attributes2', 'JPEGImages')
classes_txt_path = os.path.join(data_dir, 'Animals_with_Attributes2', 'classes.txt')
predicates_txt_path = os.path.join(data_dir, 'Animals_with_Attributes2', 'predicates.txt')
attributes_path = os.path.join(data_dir, 'Animals_with_Attributes2', 'predicate-matrix-binary.txt')
model_path = '/root/.ipython/WegnerThesis/charts_figures_etc/best_model.pth'
output_dir = '/root/.ipython/WegnerThesis/charts_figures_etc/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Hyperparameters
batch_size = 32
num_attributes = 85  # Number of attributes/predicates in AWA2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transforms
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalize with ImageNet mean and std
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------
# Custom Dataset Class
# ----------------------------
class AWA2Dataset(Dataset):
    def __init__(self, root_dir, classes_txt_path, predicates_txt_path, attributes_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Load class names
        self.classes = []
        self.class_to_idx = {}
        with open(classes_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    idx = int(parts[0]) - 1  # Indices start from 1 in classes.txt
                    cls_name = parts[1]
                    self.classes.append(cls_name)
                    self.class_to_idx[cls_name] = idx
                else:
                    print(f"[Warning] Malformed line in classes.txt: {line.strip()}")

        # Load attribute names
        self.attributes = []
        with open(predicates_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    idx = int(parts[0]) - 1  # Indices start from 1 in predicates.txt
                    attr_name = parts[1]
                    self.attributes.append(attr_name)
                else:
                    print(f"[Warning] Malformed line in predicates.txt: {line.strip()}")

        # Load attribute matrix
        self.attribute_matrix = np.loadtxt(attributes_path, dtype=int)  # Shape: (N_classes, N_attributes)

        # Build mapping from class index to attribute vector
        self.class_idx_to_attributes = {}
        for idx, cls_name in enumerate(self.classes):
            self.class_idx_to_attributes[idx] = self.attribute_matrix[idx]

        # Collect all image paths and labels
        self.image_paths = []
        self.labels = []
        self.image_classes = []  # Store class names for each image
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                print(f"[Warning] Class directory '{cls_dir}' not found, skipping...")
                continue
            class_idx = self.class_to_idx[cls_name]
            label = self.attribute_matrix[class_idx]
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(label)
                    self.image_classes.append(cls_name)
                else:
                    print(f"[Warning] '{img_path}' is not a file, skipping...")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[Error] Could not open image {img_path}: {e}")
            # Return a dummy image or skip
            image = Image.new('RGB', (224, 224))
        label = self.labels[idx]
        label = torch.from_numpy(label).float()  # Convert label to float tensor
        if self.transform:
            image = self.transform(image)
        img_name = os.path.basename(img_path)
        class_name = self.image_classes[idx]
        return image, label, img_name, class_name

# ----------------------------
# Load Validation Data
# ----------------------------
print("Loading dataset...")
full_dataset = AWA2Dataset(root_dir=images_dir,
                           classes_txt_path=classes_txt_path,
                           predicates_txt_path=predicates_txt_path,
                           attributes_path=attributes_path,
                           transform=data_transform)
print(f"Total images in dataset: {len(full_dataset)}")

# Split dataset into training and validation sets
from sklearn.model_selection import train_test_split

train_indices, val_indices = train_test_split(list(range(len(full_dataset))), test_size=0.2, random_state=42)
val_subset = torch.utils.data.Subset(full_dataset, val_indices)

print(f"Number of validation samples: {len(val_subset)}")

# Data loader
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

# ----------------------------
# Load the Best Model
# ----------------------------
print("Loading the best model...")
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_attributes)  # Adjust the final layer for multi-label classification
model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=device))
model = model.to(device)
model.eval()

# ----------------------------
# Run Inference and Calculate Absolute Errors
# ----------------------------
print("Running inference on validation set...")
attribute_names = full_dataset.attributes  # List of attribute names

abs_errors = []  # To store absolute errors
val_image_names = []
val_class_names = []

with torch.no_grad():
    for images, labels, img_names, class_names in tqdm(val_loader, desc="Processing"):
        images = images.to(device)
        outputs = model(images)
        outputs = torch.sigmoid(outputs).cpu()
        labels = labels.cpu()

        # Binarize outputs at threshold 0.5
        binarized_outputs = (outputs > 0.5).float()

        # Calculate absolute errors
        abs_error = torch.abs(binarized_outputs - labels)
        abs_errors.append(abs_error)

        val_image_names.extend(img_names)
        val_class_names.extend(class_names)

# Stack all absolute errors
abs_errors = torch.cat(abs_errors, dim=0)  # Shape: (num_samples, num_attributes)

# ----------------------------
# Save Absolute Errors to CSV
# ----------------------------
output_csv_path = os.path.join(output_dir, 'validation_attribute_errors.csv')
print(f"Saving attribute errors to {output_csv_path}...")

# Prepare CSV header
header = ['ImageName', 'ClassName'] + attribute_names

# Write to CSV
with open(output_csv_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(header)
    for idx in range(len(val_image_names)):
        img_name = val_image_names[idx]
        class_name = val_class_names[idx]
        errors = abs_errors[idx].int().numpy().tolist()
        row = [img_name, class_name] + errors
        csvwriter.writerow(row)

print("Done.")

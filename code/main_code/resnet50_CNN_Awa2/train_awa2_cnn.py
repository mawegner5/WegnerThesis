#!/usr/bin/env python3
# train_awa2_cnn.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----------------------------
# Define the Soft Jaccard Loss Function
# ----------------------------
class SoftJaccardLoss(nn.Module):
    def __init__(self):
        super(SoftJaccardLoss, self).__init__()
    
    def forward(self, outputs, targets):
        # Apply Sigmoid for multi-label classification
        sigmoid_outputs = torch.sigmoid(outputs)
        intersection = (targets * sigmoid_outputs).sum(dim=1)
        union = (targets + sigmoid_outputs).sum(dim=1) - intersection + 1e-10
        soft_jaccard = 1 - (intersection / union)
        return soft_jaccard.mean()

# ----------------------------
# Define the Early Stopping Class
# ----------------------------
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0.0, path='best_model.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_jaccard = 0.0

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation score increases.'''
        if self.verbose:
            print(f'Validation Jaccard increased to {score:.4f}. Saving model...')
        torch.save(model.state_dict(), self.path)
        self.best_jaccard = score

# ----------------------------
# Define the AWA2Dataset Class
# ----------------------------
class AWA2Dataset(Dataset):
    def __init__(self, class_dirs, csv_path, transform=None):
        """
        Args:
            class_dirs (list): List of directories, each corresponding to a class containing images.
            csv_path (str): Path to the predicate_matrix_with_labels.csv file.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.transform = transform

        # Load the CSV file
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded CSV with shape: {df.shape}")
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")

        # Create a mapping from class name to attributes
        self.class_to_attributes = {}
        for _, row in df.iterrows():
            class_name = row['class'].strip()
            attributes = row.drop('class').values.astype(int)
            self.class_to_attributes[class_name] = attributes

        # Collect all image paths and labels
        self.image_paths = []
        self.labels = []
        for class_dir in class_dirs:
            if not os.path.isdir(class_dir):
                print(f"Skipping non-directory: {class_dir}")
                continue
            class_name = os.path.basename(class_dir)
            if class_name not in self.class_to_attributes:
                print(f"Class name '{class_name}' not found in CSV. Skipping.")
                continue
            label = self.class_to_attributes[class_name]

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
            # Skip corrupted images by returning a black image
            print(f"Error loading image {img_path}: {e}. Using a black image instead.")
            image = Image.new('RGB', (224, 224))
        label = self.labels[idx]
        label = torch.from_numpy(label).float()
        if self.transform:
            image = self.transform(image)
        return image, label

# ----------------------------
# Paths and Hyperparameters
# ----------------------------
# Paths
data_dir = '/root/.ipython/WegnerThesis/animals_with_attributes/animals_w_att_data/'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validate')
csv_path = os.path.join(data_dir, 'Animals_with_Attributes2', 'predicate_matrix_with_labels.csv')
output_dir = '/root/.ipython/WegnerThesis/charts_figures_etc/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Hyperparameters
num_epochs = 100  # Set to a higher number; early stopping will halt training when appropriate
num_attributes = 85  # Number of attributes/predicates in AWA2
batch_size = 32
learning_rate = 1e-3  # Increased initial learning rate
weight_decay = 1e-5
patience = 5  # Early stopping patience
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ----------------------------
# Data Preparation
# ----------------------------
# Read class names
def read_classes(file_path):
    if not os.path.exists(file_path):
        print(f"ERROR: Class list file does not exist: {file_path}")
        exit(1)
    with open(file_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

# Read train and validation class names
train_classes_txt = os.path.join(data_dir, 'Animals_with_Attributes2', 'trainclasses.txt')
val_classes_txt = os.path.join(data_dir, 'Animals_with_Attributes2', 'valclasses.txt')

print(f"Validation classes file path: {val_classes_txt}")
print(f"Does valclasses.txt exist? {'Yes' if os.path.exists(val_classes_txt) else 'No'}")

train_classes = read_classes(train_classes_txt)
val_classes = read_classes(val_classes_txt)

print(f"Training classes ({len(train_classes)}): {train_classes}")
print(f"Validation classes ({len(val_classes)}): {val_classes}")

# Define transforms
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create datasets
train_class_dirs = [os.path.join(train_dir, cls) for cls in train_classes]
val_class_dirs = [os.path.join(val_dir, cls) for cls in val_classes]

train_dataset = AWA2Dataset(
    class_dirs=train_class_dirs,
    csv_path=csv_path,
    transform=data_transform
)

val_dataset = AWA2Dataset(
    class_dirs=val_class_dirs,
    csv_path=csv_path,
    transform=data_transform
)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")

# Create data loaders with num_workers=0 to avoid shared memory issues
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# ----------------------------
# Initialize the Model
# ----------------------------
# Use pretrained weights for better performance
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_attributes)  # Modify final layer
model = model.to(device)

# Define loss function and optimizer
criterion = SoftJaccardLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning Rate Scheduler: Reduce LR by a factor of 0.1 if validation loss doesn't improve for 'patience' epochs
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)

# ----------------------------
# Initialize Early Stopping
# ----------------------------
early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0.0, path=os.path.join(output_dir, 'best_resnet50_awa2.pth'))

# ----------------------------
# Initialize Metrics Tracking
# ----------------------------
training_losses = []
validation_losses = []
validation_jaccard = []

# ----------------------------
# Training Loop
# ----------------------------
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc=f'Epoch {epoch}/{num_epochs}')
    for batch_idx, (images, labels) in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 50 == 0:
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(train_loader)
    training_losses.append(epoch_loss)
    print(f"\nEpoch [{epoch}/{num_epochs}] Training Loss: {epoch_loss:.4f}")

    # Validation Phase
    model.eval()
    val_running_loss = 0.0
    val_all_targets = []
    val_all_outputs = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation', leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            # Apply Sigmoid as per Soft Jaccard definition
            outputs = torch.sigmoid(outputs).cpu()
            labels = labels.cpu()
            val_all_outputs.extend(outputs)
            val_all_targets.extend(labels)

    val_epoch_loss = val_running_loss / len(val_loader)
    validation_losses.append(val_epoch_loss)
    print(f"Epoch [{epoch}/{num_epochs}] Validation Loss: {val_epoch_loss:.4f}")

    # Calculate Jaccard Score (Intersection over Union)
    val_all_outputs = torch.stack(val_all_outputs)
    val_all_targets = torch.stack(val_all_targets)
    val_binarized_outputs = (val_all_outputs > 0.5).float()

    # Compute Jaccard Score for each sample
    intersection = (val_binarized_outputs * val_all_targets).sum(dim=1)
    union = (val_binarized_outputs + val_all_targets).sum(dim=1) - intersection + 1e-10
    jaccard_scores = (intersection / union).numpy()
    mean_jaccard = jaccard_scores.mean()
    validation_jaccard.append(mean_jaccard)

    print(f"Epoch [{epoch}/{num_epochs}] Validation Jaccard Score: {mean_jaccard:.4f}")

    # Step the scheduler based on validation loss
    scheduler.step(val_epoch_loss)

    # Early Stopping
    early_stopping(mean_jaccard, model)

    if early_stopping.early_stop:
        print("Early stopping triggered. Stopping training.")
        break

# ----------------------------
# Plot Training and Validation Metrics
# ----------------------------
epochs_range = range(1, len(training_losses) + 1)

# Plot Loss
plt.figure(figsize=(10,5))
plt.plot(epochs_range, training_losses, label='Training Loss', marker='o')
plt.plot(epochs_range, validation_losses, label='Validation Loss', marker='x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
plt.close()

# Plot Jaccard Score
plt.figure(figsize=(10,5))
plt.plot(epochs_range, validation_jaccard, label='Validation Jaccard Score', color='green', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Jaccard Score')
plt.title('Validation Jaccard Score Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'jaccard_score_plot.png'))
plt.close()

# ----------------------------
# Save the Final Model
# ----------------------------
final_model_path = os.path.join(output_dir, 'resnet50_awa2_final.pth')
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved to {final_model_path}")

# ----------------------------
# Save Metrics Data
# ----------------------------
import csv

metrics_path = os.path.join(output_dir, 'training_metrics.csv')
with open(metrics_path, 'w', newline='') as csvfile:
    fieldnames = ['epoch', 'training_loss', 'validation_loss', 'validation_jaccard']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for epoch_num in range(len(training_losses)):
        writer.writerow({
            'epoch': epoch_num + 1,
            'training_loss': training_losses[epoch_num],
            'validation_loss': validation_losses[epoch_num],
            'validation_jaccard': validation_jaccard[epoch_num]
        })

print(f"Training metrics saved to {metrics_path}")

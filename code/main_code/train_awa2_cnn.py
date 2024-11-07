import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm  # For progress bars

# ----------------------------
# Hyperparameters and Settings
# ----------------------------
# Paths
data_dir = '/root/.ipython/WegnerThesis/animals_with_attributes/animals_w_att_data/'
images_dir = os.path.join(data_dir, 'Animals_with_Attributes2', 'JPEGImages')
classes_txt_path = os.path.join(data_dir, 'Animals_with_Attributes2', 'classes.txt')
predicates_txt_path = os.path.join(data_dir, 'Animals_with_Attributes2', 'predicates.txt')
attributes_path = os.path.join(data_dir, 'Animals_with_Attributes2', 'predicate-matrix-binary.txt')
output_dir = '/root/.ipython/WegnerThesis/charts_figures_etc/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Hyperparameters
batch_size = 32
num_epochs = 100  # Set to a large number; early stopping will halt training when appropriate
learning_rate = 0.0001  # Smaller learning rate for fine-tuning
num_attributes = 85  # Number of attributes/predicates in AWA2
patience = 5  # For early stopping
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transforms
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Data augmentation
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
# Data Preparation
# ----------------------------
# Load dataset
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
train_subset = torch.utils.data.Subset(full_dataset, train_indices)
val_subset = torch.utils.data.Subset(full_dataset, val_indices)

print(f"Number of training samples: {len(train_subset)}")
print(f"Number of validation samples: {len(val_subset)}")

# Data loaders
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

# ----------------------------
# Model Setup
# ----------------------------
# Define the model: ResNet50 pretrained on ImageNet
print("Initializing model...")
model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_attributes)  # Adjust the final layer for multi-label classification
model = model.to(device)

# Define the loss function: Binary Cross-Entropy Loss with Logits
criterion = nn.BCEWithLogitsLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

# ----------------------------
# Training Loop with Early Stopping
# ----------------------------
print("Starting training...")
best_val_loss = float('inf')
trigger_times = 0

train_losses = []
val_losses = []
train_jaccard_scores = []
val_jaccard_scores = []

for epoch in range(num_epochs):
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")
    # Training Phase
    model.train()
    running_loss = 0.0
    all_targets = []
    all_outputs = []

    for images, labels, _, _ in tqdm(train_loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Collect outputs and targets for metric calculation
        all_outputs.append(torch.sigmoid(outputs).detach().cpu())
        all_targets.append(labels.cpu())

    epoch_loss = running_loss / len(train_subset)
    train_losses.append(epoch_loss)

    # Calculate Jaccard score for the training epoch
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    # Binarize outputs at threshold 0.5
    binarized_outputs = (all_outputs > 0.5).float()
    jaccard = jaccard_score(all_targets.numpy(), binarized_outputs.numpy(), average='samples')
    train_jaccard_scores.append(jaccard)

    print(f"Training Loss: {epoch_loss:.4f}, Training Jaccard Score: {jaccard:.4f}")

    # Validation Phase
    model.eval()
    val_running_loss = 0.0
    val_all_targets = []
    val_all_outputs = []
    val_image_names = []
    val_class_names = []

    with torch.no_grad():
        for images, labels, img_names, class_names in tqdm(val_loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)

            # Collect outputs and targets for metric calculation
            outputs = torch.sigmoid(outputs).cpu()
            labels = labels.cpu()
            val_all_outputs.extend(outputs)
            val_all_targets.extend(labels)
            val_image_names.extend(img_names)
            val_class_names.extend(class_names)

    val_epoch_loss = val_running_loss / len(val_subset)
    val_losses.append(val_epoch_loss)

    # Stack outputs and targets
    val_all_outputs = torch.stack(val_all_outputs)
    val_all_targets = torch.stack(val_all_targets)

    # Calculate Jaccard score for the validation epoch
    # Binarize outputs at threshold 0.5
    val_binarized_outputs = (val_all_outputs > 0.5).float()
    val_jaccard = jaccard_score(val_all_targets.numpy(), val_binarized_outputs.numpy(), average='samples')
    val_jaccard_scores.append(val_jaccard)

    print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Jaccard Score: {val_jaccard:.4f}")

    # Scheduler step
    scheduler.step(val_epoch_loss)

    # Early Stopping Check
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        trigger_times = 0
        # Save the best model
        torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
        print("Validation loss decreased. Saving model...")
    else:
        trigger_times += 1
        print(f"EarlyStopping counter: {trigger_times} out of {patience}")
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break

print("\nTraining complete.")

# ----------------------------
# Save Training Plots and Predictions
# ----------------------------
# Plotting training and validation loss
epochs_range = range(1, len(train_losses) + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, train_losses, label='Training Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, 'training_validation_loss.png'))
plt.close()

# Plotting training and validation Jaccard Score
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, train_jaccard_scores, label='Training Jaccard Score')
plt.plot(epochs_range, val_jaccard_scores, label='Validation Jaccard Score')
plt.xlabel('Epoch')
plt.ylabel('Jaccard Score')
plt.title('Training and Validation Jaccard Score')
plt.legend()
plt.savefig(os.path.join(output_dir, 'training_validation_jaccard.png'))
plt.close()

# Load the best model for generating predictions
model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))

# Create CSV with predictions
attribute_names = full_dataset.attributes  # List of attribute names

print("Generating predictions on validation set...")
model.eval()
val_all_outputs = []
val_image_names = []
val_class_names = []

with torch.no_grad():
    for images, labels, img_names, class_names in tqdm(val_loader, desc="Predicting", leave=False):
        images = images.to(device)
        outputs = model(images)
        outputs = torch.sigmoid(outputs).cpu()
        val_all_outputs.extend(outputs)
        val_image_names.extend(img_names)
        val_class_names.extend(class_names)

# For each image, get the predicted attributes
with open(os.path.join(output_dir, 'val_predictions.csv'), 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['ImageName', 'ClassName', 'PredictedAttributes'])
    for idx in range(len(val_image_names)):
        img_name = val_image_names[idx]
        class_name = val_class_names[idx]
        outputs = val_all_outputs[idx]
        binarized_outputs = (outputs > 0.5).int().numpy()
        predicted_attrs = [attribute_names[i] for i in range(len(attribute_names)) if binarized_outputs[i] == 1]
        predicted_attrs_str = ','.join(predicted_attrs)
        csvwriter.writerow([img_name, class_name, predicted_attrs_str])

print(f"Validation predictions saved to {os.path.join(output_dir, 'val_predictions.csv')}")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
import os
import numpy as np
import pandas as pd
import time
import copy
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# --------------------------
# User-Modifiable Parameters
# --------------------------

# Data directories
data_dir = '/remote_home/WegnerThesis/animals_with_attributes/animals_w_att_data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validate')

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Number of worker processes for data loading
num_workers = 0  # Set to 0 to avoid DataLoader worker issues

# Test parameters
batch_size = 16  # Small batch size for testing
num_epochs = 1   # Only one epoch for testing
subset_size = 64  # Number of samples to use for testing
threshold = 0.5   # Threshold for binary predictions

# --------------------------
#       End of User Settings
# --------------------------

# Load attribute names and class-attribute matrix from CSV
attributes_csv_path = os.path.join(data_dir, 'predicate_matrix_with_labels.csv')
attributes_df = pd.read_csv(attributes_csv_path, index_col=0)

# Adjust class names to match directory names (replace spaces with '+')
attributes_df.index = attributes_df.index.str.replace(' ', '+')
attribute_names = attributes_df.columns.tolist()
classes = attributes_df.index.tolist()

num_attributes = len(attribute_names)

# Custom dataset to include attributes and image names
class AwA2Dataset(Dataset):
    def __init__(self, root_dir, phase, transform=None):
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform
        self.samples = []
        self.attributes = []

        # Map class names to attributes
        self.class_to_attributes = attributes_df.to_dict(orient='index')

        # Prepare the dataset
        self._prepare_dataset()

    def _prepare_dataset(self):
        phase_dir = os.path.join(self.root_dir, self.phase)
        for class_name in os.listdir(phase_dir):
            class_dir = os.path.join(phase_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            if class_name not in self.class_to_attributes:
                print(f"Warning: Class {class_name} not found in attribute list.")
                continue
            class_attributes = np.array(list(self.class_to_attributes[class_name].values()), dtype=np.float32)

            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, img_name))  # Store both path and name
                self.attributes.append(class_attributes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, img_name = self.samples[idx]
        attributes = self.attributes[idx]
        try:
            image = datasets.folder.default_loader(img_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Create a blank image in case of error
            image = Image.new('RGB', (224, 224))
        if self.transform is not None:
            image = self.transform(image)
        attributes = torch.FloatTensor(attributes)
        return image, attributes, img_name  # Return image name

# Define soft Jaccard loss function
class SoftJaccardLoss(nn.Module):
    def __init__(self):
        super(SoftJaccardLoss, self).__init__()

    def forward(self, outputs, targets):
        eps = 1e-7
        outputs = torch.sigmoid(outputs)  # Ensuring sigmoid activation
        intersection = (outputs * targets).sum(dim=1)
        union = (outputs + targets - outputs * targets).sum(dim=1)
        loss = 1 - (intersection + eps) / (union + eps)
        return loss.mean()

def test_model():
    # Data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # Normalization values are standard for ImageNet
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # Load datasets
    full_dataset = AwA2Dataset(data_dir, 'train', transform=data_transforms['train'])

    # Create a subset of the dataset for testing
    indices = np.random.choice(len(full_dataset), subset_size, replace=False)
    test_dataset = Subset(full_dataset, indices)

    # Data loader
    dataloader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers, pin_memory=False)

    # Load ResNet50 model without pre-trained weights
    from torchvision.models import resnet50

    model = resnet50(weights=None)

    # Add dropout for regularization
    # Replace the fully connected layer with a custom layer that includes dropout
    class ResNet50WithDropout(nn.Module):
        def __init__(self, original_model, dropout_rate=0.5):
            super(ResNet50WithDropout, self).__init__()
            self.features = nn.Sequential(*list(original_model.children())[:-1])  # Exclude the last fc layer
            self.dropout = nn.Dropout(p=dropout_rate)
            self.fc = nn.Linear(original_model.fc.in_features, num_attributes)

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.fc(x)
            return x

    model = ResNet50WithDropout(model, dropout_rate=0.5)
    model = model.to(device)

    # Instantiate the loss function
    criterion = SoftJaccardLoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    model.train()
    print("\nStarting test training loop...")
    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            running_loss = 0.0
            running_jaccard = 0.0

            progress_bar = tqdm(enumerate(dataloader), desc=f"Epoch {epoch+1}",
                                total=len(dataloader), unit='batch')

            for batch_idx, (inputs, labels, img_names) in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.sigmoid(outputs)
                preds_binary = (preds >= threshold).float()

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Collect predictions and labels for Jaccard score calculation
                preds_np = preds_binary.detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()

                # Update running Jaccard score
                batch_jaccard = jaccard_score(labels_np, preds_np, average='samples', zero_division=0)
                running_jaccard += batch_jaccard * inputs.size(0)

                # Statistics
                running_loss += loss.item() * inputs.size(0)

                # Update progress bar
                batch_loss = running_loss / ((batch_idx + 1) * inputs.size(0))
                batch_jaccard_avg = running_jaccard / ((batch_idx + 1) * inputs.size(0))
                progress_bar.set_postfix({'Loss': f'{batch_loss:.4f}', 'Jaccard': f'{batch_jaccard_avg:.4f}'})

            epoch_loss = running_loss / subset_size
            epoch_jaccard = running_jaccard / subset_size

            print(f"\nEpoch {epoch+1} Loss: {epoch_loss:.4f} Jaccard: {epoch_jaccard:.4f}")

        print("\nTest training loop completed successfully.")

    except RuntimeError as e:
        print(f"RuntimeError during testing: {e}")
        print("The error suggests that the modifications may not have fully resolved the issue.")
        print("Consider further adjustments or checking system resources.")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")

if __name__ == '__main__':
    test_model()

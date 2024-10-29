import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from awa2dataset import AWA2Dataset  # Import the dataset class you defined
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Directories
data_dir = '/root/.ipython/WegnerThesis/animals_with_attributes/animals_w_att_data/'
checkpoint_dir = '/root/.ipython/WegnerThesis/checkpoints/'
plots_dir = '/root/.ipython/WegnerThesis/charts_figures_etc/'
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Parameters
batch_size = 32
num_epochs = 100
learning_rate = 1e-4
early_stop_patience = 10
num_attributes = 85  # Number of attributes/predicates in AWA2

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = AWA2Dataset(root_dir=os.path.join(data_dir, 'train'), transform=train_transform)
val_dataset = AWA2Dataset(root_dir=os.path.join(data_dir, 'validate'), transform=val_transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load pre-trained model
model = models.resnet101(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_attributes)  # Adjust final layer for 85 attributes
model = model.to(device)

# Freeze earlier layers if desired (optional)
# for name, param in model.named_parameters():
#     if 'fc' not in name:
#         param.requires_grad = False

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_loader), epochs=num_epochs)

# Training loop with early stopping
best_val_loss = float('inf')
early_stop_counter = 0

# Lists to store losses for plotting
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
        images = images.to(device)
        labels = labels.to(device).float()  # Ensure labels are float tensors

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * images.size(0)

    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
            images = images.to(device)
            labels = labels.to(device).float()

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)

    epoch_val_loss = val_running_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}')

    # Checkpointing
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_model_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f"Checkpoint saved at {best_model_path}")
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    # Early stopping
    if early_stop_counter >= early_stop_patience:
        print("Early stopping triggered")
        break

# Save final model
final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
torch.save(model.state_dict(), final_model_path)
print(f"Training Complete. Final model saved at {final_model_path}")

# Plotting training and validation loss
epochs_range = range(1, len(train_losses) + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs_range, train_losses, label='Training Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plot_filename = os.path.join(plots_dir, 'training_validation_loss.png')
plt.savefig(plot_filename)
plt.close()
print(f"Loss plot saved at {plot_filename}")

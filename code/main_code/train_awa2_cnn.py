import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from awa2dataset import AWA2Dataset  # Import the dataset class you defined
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

# Define paths and parameters
data_dir = '/root/.ipython/WegnerThesis/animals_with_attributes/animals_w_att_data/train'
checkpoint_dir = '/root/.ipython/WegnerThesis/checkpoints/'
os.makedirs(checkpoint_dir, exist_ok=True)
batch_size = 32
num_epochs = 100
learning_rate = 1e-4
early_stop_patience = 10

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset and dataloader
train_dataset = AWA2Dataset(root_dir=data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Load pre-trained model
model = models.resnet101(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_dataset.classes))  # Adjust final layer for AWA2 classes
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Freeze earlier layers, fine-tune last layers
for name, param in model.named_parameters():
    if 'fc' not in name:  # Freeze all layers except the last fully connected layer
        param.requires_grad = False

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
scheduler = OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_loader), epochs=num_epochs)

# Training loop with early stopping
best_loss = float('inf')
early_stop_counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        images, labels = images.to('cuda'), labels.to('cuda')
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    # Checkpointing
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
        print("Checkpoint saved!")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
    
    # Early stopping
    if early_stop_counter >= early_stop_patience:
        print("Early stopping triggered")
        break

print("Training Complete. Best model saved at", os.path.join(checkpoint_dir, 'best_model.pth'))

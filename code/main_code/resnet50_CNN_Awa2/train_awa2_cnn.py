import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
import time
import copy
from sklearn.metrics import classification_report, confusion_matrix, jaccard_score
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm  # Progress bar

# Define data directories
data_dir = '/root/.ipython/WegnerThesis/animals_with_attributes/animals_w_att_data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validate')
test_dir = os.path.join(data_dir, 'test')

# Data transformations (no augmentation for initial runs)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalization values are standard for ImageNet
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validate': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load datasets
image_datasets = {
    'train': datasets.ImageFolder(train_dir, data_transforms['train']),
    'validate': datasets.ImageFolder(val_dir, data_transforms['validate']),
}

# Compute class weights to handle class imbalance
train_targets = [label for _, label in image_datasets['train'].imgs]
class_counts = np.bincount(train_targets)
class_weights = 1. / class_counts
class_weights = torch.FloatTensor(class_weights)

# Data loaders
batch_size = 16

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=1),
    'validate': DataLoader(image_datasets['validate'], batch_size=batch_size, shuffle=False, num_workers=1),
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validate']}
class_names = image_datasets['train'].classes

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load ResNet50 model without pre-trained weights
from torchvision.models import resnet50

model = resnet50(weights=None)

# Modify the final layer to match the number of classes
num_ftrs = model.fc.in_features
num_classes = len(class_names)
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# Define soft Jaccard loss function
class SoftJaccardLoss(nn.Module):
    def __init__(self):
        super(SoftJaccardLoss, self).__init__()

    def forward(self, outputs, targets):
        eps = 1e-7
        num_classes = outputs.shape[1]
        # Convert targets to one-hot encoding
        targets_one_hot = torch.zeros_like(outputs).scatter(1, targets.unsqueeze(1), 1)
        outputs = torch.softmax(outputs, dim=1)

        intersection = (outputs * targets_one_hot).sum(dim=0)
        union = (outputs + targets_one_hot - outputs * targets_one_hot).sum(dim=0)
        jaccard = (intersection + eps) / (union + eps)
        loss = 1 - jaccard.mean()
        return loss

# Instantiate the loss function
criterion = SoftJaccardLoss()

# Define optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Define learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.verbose:
                print(f'Validation loss decreased to {val_loss:.6f}')
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                print(f'Validation loss did not improve from {self.best_loss:.6f}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('Early stopping triggered')
        else:
            if self.verbose:
                print(f'Validation loss decreased from {self.best_loss:.6f} to {val_loss:.6f}')
            self.best_loss = val_loss
            self.counter = 0

# Instantiate early stopping
early_stopping = EarlyStopping(patience=5, verbose=True)

# Output directory for saving predictions and reports
output_dir = '/root/.ipython/WegnerThesis/charts_figures_etc'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=1):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_jaccard = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validate']:
            if phase == 'train':
                model.train()  # Training mode
            else:
                model.eval()   # Evaluation mode
                val_predictions = []
                val_labels = []

            running_loss = 0.0
            running_corrects = 0

            progress_bar = tqdm(enumerate(dataloaders[phase]), desc=f"{phase.capitalize()} Epoch {epoch+1}", total=len(dataloaders[phase]), unit='batch')

            # Iterate over data
            for batch_idx, (inputs, labels) in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimization
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        # Collect predictions and labels for validation
                        val_predictions.extend(preds.cpu().numpy())
                        val_labels.extend(labels.cpu().numpy())

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Calculate batch loss and accuracy
                batch_loss = running_loss / ((batch_idx + 1) * inputs.size(0))
                batch_acc = running_corrects.double() / ((batch_idx + 1) * inputs.size(0))

                # Update progress bar
                progress_bar.set_postfix({'Loss': f'{batch_loss:.4f}', 'Acc': f'{batch_acc:.4f}'})

            # Calculate epoch loss and accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'validate':
                # Calculate Jaccard score
                jaccard = jaccard_score(val_labels, val_predictions, average='macro')
                print(f'\n{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Jaccard: {jaccard:.4f}')
                scheduler.step(epoch_loss)
                early_stopping(epoch_loss)

                # Deep copy the model if it has the best Jaccard score
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_model_wts = copy.deepcopy(model.state_dict())

                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            else:
                print(f'\n{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if early_stopping.early_stop:
            break

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Jaccard Score: {best_jaccard:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Save validation predictions and labels to file
    save_predictions(val_predictions, val_labels, iteration)

    return model

def save_predictions(predictions, labels, iteration):
    # Create a DataFrame
    df = pd.DataFrame({'Predicted': predictions, 'Actual': labels})

    # Map class indices to class_names
    df['Predicted_Class'] = df['Predicted'].apply(lambda x: class_names[x])
    df['Actual_Class'] = df['Actual'].apply(lambda x: class_names[x])

    # Generate timestamp for filename
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'predictions_iteration{iteration}_{timestamp}.csv'

    # Save to specified directory
    output_path = os.path.join(output_dir, filename)

    df.to_csv(output_path, index=False)
    print(f'Validation predictions saved to {output_path}')

    # Generate classification report
    report = classification_report(labels, predictions, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_filename = f'classification_report_iteration{iteration}_{timestamp}.csv'
    report_output_path = os.path.join(output_dir, report_filename)
    report_df.to_csv(report_output_path)
    print(f'Classification report saved to {report_output_path}')

    # Generate confusion matrix
    cm = confusion_matrix(labels, predictions)
    cm_filename = f'confusion_matrix_iteration{iteration}_{timestamp}.png'
    cm_output_path = os.path.join(output_dir, cm_filename)
    plot_confusion_matrix(cm, class_names, cm_output_path)
    print(f'Confusion matrix saved to {cm_output_path}')

def plot_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    # Threshold for text color
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Define iteration number
iteration = 1  # Update this accordingly for each run

# Train the model
num_epochs = 1  # Change this manually for longer runs
model = train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)

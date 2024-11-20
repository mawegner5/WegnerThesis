import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
import time
import copy
from sklearn.metrics import classification_report, jaccard_score
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm

# --------------------------
# User-Modifiable Parameters
# --------------------------

# Data directories
data_dir = '/root/.ipython/WegnerThesis/animals_with_attributes/animals_w_att_data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validate')
test_dir = os.path.join(data_dir, 'test')

# Model configuration
num_epochs = 120            # Adjust as needed
batch_size = 16             # Adjust based on your GPU memory
learning_rate = 0.0001      # Reduced learning rate
num_workers = 1             # Number of worker processes for data loading
iteration = 5               # For naming outputs

# Output directory for saving predictions and reports
output_dir = '/root/.ipython/WegnerThesis/charts_figures_etc'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Path to the model performance summary file
performance_summary_path = os.path.join(output_dir, 'model_performance_summary.csv')

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --------------------------
#       End of User Settings
# --------------------------

# Load attribute names and class-attribute matrix from CSV
attributes_csv_path = os.path.join(data_dir, 'Animals_with_Attributes2', 'predicate_matrix_with_labels.csv')
attributes_df = pd.read_csv(attributes_csv_path, index_col=0)
attributes = attributes_df.values  # Convert DataFrame to NumPy array
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

        # Map class names to indices
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        # Prepare the dataset
        self._prepare_dataset()

    def _prepare_dataset(self):
        phase_dir = os.path.join(self.root_dir, self.phase)
        for class_name in os.listdir(phase_dir):
            class_dir = os.path.join(phase_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            if class_name not in self.class_to_idx:
                print(f"Warning: Class {class_name} not found in class list.")
                continue
            class_idx = self.class_to_idx[class_name]
            class_attributes = attributes[class_idx]

            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, img_name))  # Store both path and name
                self.attributes.append(class_attributes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, img_name = self.samples[idx]
        attributes = self.attributes[idx]
        image = datasets.folder.default_loader(img_path)
        if self.transform is not None:
            image = self.transform(image)
        attributes = torch.FloatTensor(attributes)
        return image, attributes, img_name  # Return image name

# Data transformations with data augmentation for training
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        # Normalization values are standard for ImageNet
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'validate': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # Normalization values are standard for ImageNet
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# Load datasets
datasets_dict = {
    'train': AwA2Dataset(data_dir, 'train', transform=data_transforms['train']),
    'validate': AwA2Dataset(data_dir, 'validate', transform=data_transforms['validate']),
}

# Data loaders
dataloaders = {
    'train': DataLoader(datasets_dict['train'], batch_size=batch_size,
                        shuffle=True, num_workers=num_workers),
    'validate': DataLoader(datasets_dict['validate'], batch_size=batch_size,
                           shuffle=False, num_workers=num_workers),
}

dataset_sizes = {x: len(datasets_dict[x]) for x in ['train', 'validate']}

# Load ResNet50 model without pre-trained weights
from torchvision.models import resnet50

model = resnet50(weights=None)

# Modify the final layer to match the number of attributes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_attributes)
model = model.to(device)

# Specify the model name for saving
model_name = 'resnet50'

# Define soft Jaccard loss function
class SoftJaccardLoss(nn.Module):
    def __init__(self):
        super(SoftJaccardLoss, self).__init__()

    def forward(self, outputs, targets):
        eps = 1e-7
        outputs = torch.sigmoid(outputs)
        intersection = (outputs * targets).sum(dim=1)
        union = (outputs + targets - outputs * targets).sum(dim=1)
        loss = 1 - (intersection + eps) / (union + eps)
        return loss.mean()

# Instantiate the loss function
criterion = SoftJaccardLoss()

# Define optimizer (switched to Adam)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Define learning rate scheduler (CosineAnnealingWarmRestarts)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# Early stopping class with increased patience
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
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

# Instantiate early stopping with increased patience
early_stopping = EarlyStopping(patience=10, verbose=True)

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_jaccard = 0.0

    # Variables to store best validation results
    best_val_predictions = None
    best_val_labels = None
    best_val_img_names = None

    # Lists to store loss and jaccard scores
    train_losses = []
    val_losses = []
    train_jaccards = []
    val_jaccards = []

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
                val_img_names = []

            running_loss = 0.0
            running_jaccard = 0.0

            progress_bar = tqdm(enumerate(dataloaders[phase]), desc=f"{phase.capitalize()} Epoch {epoch+1}",
                                total=len(dataloaders[phase]), unit='batch')

            # Iterate over data
            for batch_idx, (inputs, labels, img_names) in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.sigmoid(outputs)
                    preds_binary = (preds >= 0.5).float()

                    # Backward pass and optimization
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # Collect predictions and labels for Jaccard score calculation
                    preds_np = preds_binary.detach().cpu().numpy()
                    labels_np = labels.detach().cpu().numpy()

                    # Update running Jaccard score
                    batch_jaccard = jaccard_score(labels_np, preds_np, average='samples')
                    running_jaccard += batch_jaccard * inputs.size(0)

                    if phase == 'validate':
                        val_predictions.append(preds_np)
                        val_labels.append(labels_np)
                        val_img_names.extend(img_names)

                # Statistics
                running_loss += loss.item() * inputs.size(0)

                # Update progress bar
                batch_loss = running_loss / ((batch_idx + 1) * inputs.size(0))
                batch_jaccard_avg = running_jaccard / ((batch_idx + 1) * inputs.size(0))
                progress_bar.set_postfix({'Loss': f'{batch_loss:.4f}', 'Jaccard': f'{batch_jaccard_avg:.4f}'})

            # Calculate epoch loss and Jaccard score
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_jaccard = running_jaccard / dataset_sizes[phase]

            print(f'\n{phase.capitalize()} Loss: {epoch_loss:.4f} Jaccard: {epoch_jaccard:.4f}')

            # Store losses and jaccards
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_jaccards.append(epoch_jaccard)
                scheduler.step()  # Update learning rate
            else:
                val_losses.append(epoch_loss)
                val_jaccards.append(epoch_jaccard)

                early_stopping(epoch_loss)

                # Concatenate all predictions and labels
                val_predictions = np.vstack(val_predictions)
                val_labels = np.vstack(val_labels)

                # Update best model if validation Jaccard improved
                if epoch_jaccard > best_jaccard:
                    best_jaccard = epoch_jaccard
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_val_predictions = val_predictions
                    best_val_labels = val_labels
                    best_val_img_names = val_img_names

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        if early_stopping.early_stop:
            break

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
    print(f'Best Validation Jaccard Score: {best_jaccard:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Save the best model weights with CNN type in filename
    model_save_path = os.path.join(output_dir, f'best_model_{model_name}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'Best model saved to {model_save_path}')

    # Save validation predictions and labels to file
    if best_val_predictions is not None and best_val_labels is not None and best_val_img_names is not None:
        save_predictions(best_val_predictions, best_val_labels, best_val_img_names, iteration)
    else:
        print("No validation predictions to save.")

    # Plot and save training curves
    plot_training_curves(train_losses, val_losses, train_jaccards, val_jaccards, iteration)

    # Save best performance to summary file
    save_performance_summary(model_name, best_jaccard, epoch_loss, time_elapsed)

    return model

def save_predictions(predictions, labels, img_names, iteration):
    # Ensure predictions and labels are NumPy arrays
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    img_names = np.asarray(img_names)

    # Generate timestamp for filename
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'predictions_{model_name}_iteration{iteration}_{timestamp}.csv'

    # Save to specified directory
    output_path = os.path.join(output_dir, filename)

    # Build DataFrame with predictions
    df_predictions = pd.DataFrame(predictions.astype(int), columns=attribute_names)
    df_predictions.insert(0, 'image_name', img_names)

    df_predictions.to_csv(output_path, index=False)
    print(f'Validation predictions saved to {output_path}')

    # Generate classification report
    # For multi-label classification, provide labels and target_names
    report = classification_report(labels.astype(int), predictions.astype(int), target_names=attribute_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_filename = f'classification_report_{model_name}_iteration{iteration}_{timestamp}.csv'
    report_output_path = os.path.join(output_dir, report_filename)
    report_df.to_csv(report_output_path)
    print(f'Classification report saved to {report_output_path}')

def plot_training_curves(train_losses, val_losses, train_jaccards, val_jaccards, iteration):
    epochs = range(1, len(train_losses) + 1)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Plot Loss
    plt.figure()
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    loss_plot_filename = f'{model_name}_training_validation_loss_{timestamp}.png'
    loss_plot_path = os.path.join(output_dir, loss_plot_filename)
    plt.savefig(loss_plot_path)
    plt.close()
    print(f'Training and validation loss plot saved to {loss_plot_path}')

    # Plot Jaccard Accuracy
    plt.figure()
    plt.plot(epochs, train_jaccards, 'b-', label='Training Jaccard Accuracy')
    plt.plot(epochs, val_jaccards, 'r-', label='Validation Jaccard Accuracy')
    plt.title(f'{model_name} Training and Validation Jaccard Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Jaccard Accuracy')
    plt.legend()
    acc_plot_filename = f'{model_name}_training_validation_jaccard_accuracy_{timestamp}.png'
    acc_plot_path = os.path.join(output_dir, acc_plot_filename)
    plt.savefig(acc_plot_path)
    plt.close()
    print(f'Training and validation accuracy plot saved to {acc_plot_path}')

def save_performance_summary(model_name, best_jaccard, best_val_loss, time_elapsed):
    # Prepare data
    data = {
        'Model': [model_name],
        'Best Validation Jaccard': [best_jaccard],
        'Best Validation Loss': [best_val_loss],
        'Training Time (s)': [int(time_elapsed)],
        'Timestamp': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    }
    df = pd.DataFrame(data)

    # Check if the summary file exists
    if os.path.exists(performance_summary_path):
        # Append to existing file
        df_existing = pd.read_csv(performance_summary_path)
        df = pd.concat([df_existing, df], ignore_index=True)
    # Save to CSV
    df.to_csv(performance_summary_path, index=False)
    print(f'Model performance summary updated at {performance_summary_path}')

# Train the model
if __name__ == '__main__':
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)

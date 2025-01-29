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
from tqdm import tqdm
import optuna
import datetime
from PIL import Image

# --------------------------
# User-Modifiable Parameters
# --------------------------

# Data directories
data_dir = '/remote_home/WegnerThesis/animals_with_attributes/animals_w_att_data'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validate')
test_dir = os.path.join(data_dir, 'test')

# Model configuration
model_name = 'convnext_tiny'    # Options: 'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'

# Training parameters
num_epochs_initial = 150
early_stopping_patience_initial = 25
batch_size = 32
learning_rate = 1e-4
weight_decay = 1e-5
optimizer_name = 'Adam'
threshold = 0.5
dropout_rate = 0.5
T_0 = 10

# Optuna hyperparameter tuning parameters
n_trials = 3
num_epochs_optuna = 50
early_stopping_patience_optuna = 10

# Output directory
output_dir = '/remote_home/WegnerThesis/charts_figures_etc'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Performance summary file
performance_summary_path = os.path.join(output_dir, 'model_performance_summary.csv')

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# DataLoader workers
num_workers = 0

# Input size for ConvNeXt
input_size = 224

# --------------------------
#       End of User Settings
# --------------------------

attributes_csv_path = os.path.join(data_dir, 'predicate_matrix_with_labels.csv')
attributes_df = pd.read_csv(attributes_csv_path, index_col=0)
attributes_df.index = attributes_df.index.str.replace(' ', '+')
attributes = attributes_df.values
attribute_names = attributes_df.columns.tolist()
classes = attributes_df.index.tolist()
num_attributes = len(attribute_names)

class AwA2Dataset(Dataset):
    def __init__(self, root_dir, phase, transform=None):
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform
        self.samples = []
        self.attributes = []
        self.class_to_attributes = attributes_df.to_dict(orient='index')
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
                self.samples.append((img_path, img_name))
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
            image = Image.new('RGB', (input_size, input_size))
        if self.transform is not None:
            image = self.transform(image)
        attributes = torch.FloatTensor(attributes)
        return image, attributes, img_name

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

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
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

def train_initial_model():
    num_epochs = num_epochs_initial
    early_stopping_patience = early_stopping_patience_initial

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'validate': transforms.Compose([
            transforms.Resize(input_size + 32),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    datasets_dict = {
        'train': AwA2Dataset(data_dir, 'train', transform=data_transforms['train']),
        'validate': AwA2Dataset(data_dir, 'validate', transform=data_transforms['validate']),
    }

    dataloaders = {
        'train': DataLoader(datasets_dict['train'], batch_size=batch_size,
                            shuffle=True, num_workers=num_workers, pin_memory=False),
        'validate': DataLoader(datasets_dict['validate'], batch_size=batch_size,
                               shuffle=False, num_workers=num_workers, pin_memory=False),
    }

    dataset_sizes = {x: len(datasets_dict[x]) for x in ['train', 'validate']}

    from torchvision.models import (convnext_tiny, convnext_small, convnext_base, convnext_large)

    if model_name == 'convnext_tiny':
        model = convnext_tiny(weights=None)
    elif model_name == 'convnext_small':
        model = convnext_small(weights=None)
    elif model_name == 'convnext_base':
        model = convnext_base(weights=None)
    elif model_name == 'convnext_large':
        model = convnext_large(weights=None)
    else:
        raise ValueError("Invalid model name.")

    num_ftrs = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        nn.Flatten(1),
        nn.LayerNorm(num_ftrs, eps=1e-6, elementwise_affine=True),
        nn.Dropout(p=dropout_rate),
        nn.Linear(num_ftrs, num_attributes),
    )
    model = model.to(device)

    criterion = SoftJaccardLoss()

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=2)
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_jaccard = 0.0
    best_val_predictions = None
    best_val_labels = None
    best_val_img_names = None

    train_losses = []
    val_losses = []
    train_jaccards = []
    val_jaccards = []

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'validate']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                val_predictions = []
                val_labels = []
                val_img_names = []

            running_loss = 0.0
            running_jaccard = 0.0

            progress_bar = tqdm(enumerate(dataloaders[phase]), desc=f"{phase.capitalize()} Epoch {epoch+1}",
                                total=len(dataloaders[phase]), unit='batch')

            try:
                for batch_idx, (inputs, labels, img_names) in progress_bar:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        preds = torch.sigmoid(outputs)
                        preds_binary = (preds >= threshold).float()

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        preds_np = preds_binary.detach().cpu().numpy()
                        labels_np = labels.detach().cpu().numpy()
                        batch_jaccard = jaccard_score(labels_np, preds_np, average='samples', zero_division=0)
                        running_jaccard += batch_jaccard * inputs.size(0)

                        if phase == 'validate':
                            val_predictions.append(preds_np)
                            val_labels.append(labels_np)
                            val_img_names.extend(img_names)

                    running_loss += loss.item() * inputs.size(0)
                    batch_loss = running_loss / ((batch_idx + 1) * inputs.size(0))
                    batch_jaccard_avg = running_jaccard / ((batch_idx + 1) * inputs.size(0))
                    progress_bar.set_postfix({'Loss': f'{batch_loss:.4f}', 'Jaccard': f'{batch_jaccard_avg:.4f}'})

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_jaccard = running_jaccard / dataset_sizes[phase]
                print(f'\n{phase.capitalize()} Loss: {epoch_loss:.4f} Jaccard: {epoch_jaccard:.4f}')

                if phase == 'train':
                    train_losses.append(epoch_loss)
                    train_jaccards.append(epoch_jaccard)
                    scheduler.step()
                else:
                    val_losses.append(epoch_loss)
                    val_jaccards.append(epoch_jaccard)
                    early_stopping(epoch_loss)
                    val_predictions = np.vstack(val_predictions)
                    val_labels = np.vstack(val_labels)

                    if epoch_jaccard > best_jaccard:
                        best_jaccard = epoch_jaccard
                        best_model_wts = copy.deepcopy(model.state_dict())
                        best_val_predictions = val_predictions
                        best_val_labels = val_labels
                        best_val_img_names = val_img_names

                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

            except RuntimeError as e:
                print(f"RuntimeError during {phase} phase: {e}")
                break

        if early_stopping.early_stop:
            break

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
    print(f'Best Validation Jaccard Score: {best_jaccard:.4f}')

    model.load_state_dict(best_model_wts)
    model_save_path = os.path.join(output_dir, f'best_model_{model_name}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'Best model saved to {model_save_path}')

    if best_val_predictions is not None and best_val_labels is not None and best_val_img_names is not None:
        save_predictions(best_val_predictions, best_val_labels, best_val_img_names, 'initial', model_name)
    else:
        print("No validation predictions to save.")

    plot_training_curves(train_losses, val_losses, train_jaccards, val_jaccards, 'initial', model_name)

    save_performance_summary(model_name, best_jaccard, epoch_loss, time_elapsed, 'initial', {
        'optimizer': optimizer_name,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'num_epochs': num_epochs,
        'dropout_rate': dropout_rate,
        'T_0': T_0,
        'threshold': threshold,
        'early_stopping_patience': early_stopping_patience
    })
    return model

def train_model(trial):
    num_epochs = num_epochs_optuna
    early_stopping_patience = early_stopping_patience_optuna
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    T_0 = trial.suggest_int('T_0', 10, 30, step=10)
    threshold = trial.suggest_float('threshold', 0.3, 0.7, step=0.05)
    dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.7, step=0.1)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'validate': transforms.Compose([
            transforms.Resize(input_size + 32),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    datasets_dict = {
        'train': AwA2Dataset(data_dir, 'train', transform=data_transforms['train']),
        'validate': AwA2Dataset(data_dir, 'validate', transform=data_transforms['validate']),
    }

    dataloaders = {
        'train': DataLoader(datasets_dict['train'], batch_size=batch_size,
                            shuffle=True, num_workers=num_workers, pin_memory=False),
        'validate': DataLoader(datasets_dict['validate'], batch_size=batch_size,
                               shuffle=False, num_workers=num_workers, pin_memory=False),
    }

    dataset_sizes = {x: len(datasets_dict[x]) for x in ['train', 'validate']}

    from torchvision.models import (convnext_tiny, convnext_small, convnext_base, convnext_large)

    if model_name == 'convnext_tiny':
        model = convnext_tiny(weights=None)
    elif model_name == 'convnext_small':
        model = convnext_small(weights=None)
    elif model_name == 'convnext_base':
        model = convnext_base(weights=None)
    elif model_name == 'convnext_large':
        model = convnext_large(weights=None)
    else:
        raise ValueError("Invalid model name.")

    num_ftrs = model.classifier[2].in_features
    model.classifier = nn.Sequential(
        nn.Flatten(1),
        nn.LayerNorm(num_ftrs, eps=1e-6, elementwise_affine=True),
        nn.Dropout(p=dropout_rate),
        nn.Linear(num_ftrs, num_attributes),
    )
    model = model.to(device)

    criterion = SoftJaccardLoss()

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=2)
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_jaccard = 0.0
    best_val_predictions = None
    best_val_labels = None
    best_val_img_names = None

    train_losses = []
    val_losses = []
    train_jaccards = []
    val_jaccards = []

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'validate']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                val_predictions = []
                val_labels = []
                val_img_names = []

            running_loss = 0.0
            running_jaccard = 0.0

            progress_bar = tqdm(enumerate(dataloaders[phase]), desc=f"{phase.capitalize()} Epoch {epoch+1}",
                                total=len(dataloaders[phase]), unit='batch')

            try:
                for batch_idx, (inputs, labels, img_names) in progress_bar:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        preds = torch.sigmoid(outputs)
                        preds_binary = (preds >= threshold).float()

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        preds_np = preds_binary.detach().cpu().numpy()
                        labels_np = labels.detach().cpu().numpy()
                        batch_jaccard = jaccard_score(labels_np, preds_np, average='samples', zero_division=0)
                        running_jaccard += batch_jaccard * inputs.size(0)

                        if phase == 'validate':
                            val_predictions.append(preds_np)
                            val_labels.append(labels_np)
                            val_img_names.extend(img_names)

                    running_loss += loss.item() * inputs.size(0)
                    batch_loss = running_loss / ((batch_idx + 1) * inputs.size(0))
                    batch_jaccard_avg = running_jaccard / ((batch_idx + 1) * inputs.size(0))
                    progress_bar.set_postfix({'Loss': f'{batch_loss:.4f}', 'Jaccard': f'{batch_jaccard_avg:.4f}'})

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_jaccard = running_jaccard / dataset_sizes[phase]
                print(f'\n{phase.capitalize()} Loss: {epoch_loss:.4f} Jaccard: {epoch_jaccard:.4f}')

                if phase == 'train':
                    train_losses.append(epoch_loss)
                    train_jaccards.append(epoch_jaccard)
                    scheduler.step()
                else:
                    val_losses.append(epoch_loss)
                    val_jaccards.append(epoch_jaccard)
                    early_stopping(epoch_loss)
                    val_predictions = np.vstack(val_predictions)
                    val_labels = np.vstack(val_labels)

                    if epoch_jaccard > best_jaccard:
                        best_jaccard = epoch_jaccard
                        best_model_wts = copy.deepcopy(model.state_dict())
                        best_val_predictions = val_predictions
                        best_val_labels = val_labels
                        best_val_img_names = val_img_names

                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

            except RuntimeError as e:
                print(f"RuntimeError during {phase} phase: {e}")
                return float('inf')

        if early_stopping.early_stop:
            break

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
    print(f'Best Validation Jaccard Score: {best_jaccard:.4f}')

    model.load_state_dict(best_model_wts)
    model_save_path = os.path.join(output_dir, f'best_model_{model_name}_trial{trial.number}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'Best model saved to {model_save_path}')

    if best_val_predictions is not None and best_val_labels is not None and best_val_img_names is not None:
        save_predictions(best_val_predictions, best_val_labels, best_val_img_names, trial.number, model_name)
    else:
        print("No validation predictions to save.")

    plot_training_curves(train_losses, val_losses, train_jaccards, val_jaccards, trial.number, model_name)
    save_performance_summary(model_name, best_jaccard, epoch_loss, time_elapsed, trial.number, trial.params)
    return epoch_loss

def save_predictions(predictions, labels, img_names, trial_number, model_name):
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    img_names = np.asarray(img_names)

    if trial_number == 'initial':
        filename = f'predictions_{model_name}.csv'
    else:
        filename = f'predictions_{model_name}_trial{trial_number}.csv'

    output_path = os.path.join(output_dir, filename)
    df_predictions = pd.DataFrame(predictions.astype(int), columns=attribute_names)
    df_predictions.insert(0, 'image_name', img_names)
    df_predictions.to_csv(output_path, index=False)
    print(f'Validation predictions saved to {output_path}')

    report = classification_report(labels.astype(int), predictions.astype(int),
                                   target_names=attribute_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    if trial_number == 'initial':
        report_filename = f'classification_report_{model_name}.csv'
    else:
        report_filename = f'classification_report_{model_name}_trial{trial_number}.csv'
    report_output_path = os.path.join(output_dir, report_filename)
    report_df.to_csv(report_output_path)
    print(f'Classification report saved to {report_output_path}')

def plot_training_curves(train_losses, val_losses, train_jaccards, val_jaccards, trial_number, model_name):
    epochs = range(1, len(train_losses) + 1)

    if trial_number == 'initial':
        loss_plot_filename = f'{model_name}_training_validation_loss.png'
        acc_plot_filename = f'{model_name}_training_validation_jaccard_accuracy.png'
    else:
        loss_plot_filename = f'{model_name}_training_validation_loss_trial{trial_number}.png'
        acc_plot_filename = f'{model_name}_training_validation_jaccard_accuracy_trial{trial_number}.png'

    plt.figure()
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    loss_plot_path = os.path.join(output_dir, loss_plot_filename)
    plt.savefig(loss_plot_path)
    plt.close()
    print(f'Training and validation loss plot saved to {loss_plot_path}')

    plt.figure()
    plt.plot(epochs, train_jaccards, 'b-', label='Training Jaccard Accuracy')
    plt.plot(epochs, val_jaccards, 'r-', label='Validation Jaccard Accuracy')
    plt.title(f'{model_name} Training and Validation Jaccard Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Jaccard Accuracy')
    plt.legend()
    acc_plot_path = os.path.join(output_dir, acc_plot_filename)
    plt.savefig(acc_plot_path)
    plt.close()
    print(f'Training and validation accuracy plot saved to {acc_plot_path}')

def save_performance_summary(model_name, best_jaccard, best_val_loss, time_elapsed, trial_number, params):
    data = {
        'Trial': [trial_number],
        'Model': [model_name],
        'Best Validation Jaccard': [best_jaccard],
        'Best Validation Loss': [best_val_loss],
        'Training Time (s)': [int(time_elapsed)],
        'Timestamp': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    }
    for key, value in params.items():
        data[key.capitalize().replace('_', ' ')] = [value]

    df = pd.DataFrame(data)
    if os.path.exists(performance_summary_path):
        df_existing = pd.read_csv(performance_summary_path)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(performance_summary_path, index=False)
    print(f'Model performance summary updated at {performance_summary_path}')

if __name__ == '__main__':
    model = train_initial_model()
    study = optuna.create_study(direction='minimize')
    study.optimize(train_model, n_trials=n_trials)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print(f'  Trial Number: {trial.number}')
    print(f'  Loss: {trial.value}')
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

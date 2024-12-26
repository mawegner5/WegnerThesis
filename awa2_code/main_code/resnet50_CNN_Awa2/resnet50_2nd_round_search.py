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

data_dir = '/remote_home/WegnerThesis/animals_with_attributes/animals_w_att_data'
output_dir = '/remote_home/WegnerThesis/charts_figures_etc'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

performance_summary_path = os.path.join(output_dir, 'model_performance_summary.csv')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_workers = 0  # Set to 0 to avoid DataLoader worker issues

# Round 2 parameters
n_trials = 5  # Number of trials for the second round
num_epochs = 500  # Train longer to get closer to overfitting
early_stopping_patience = 50  # Large patience since we're training longer
training_jaccard_threshold = 0.90  # Train on training set only until this Jaccard is reached before validation

# --------------------------
#       End of User Settings
# --------------------------

attributes_csv_path = os.path.join(data_dir, 'predicate_matrix_with_labels.csv')
attributes_df = pd.read_csv(attributes_csv_path, index_col=0)
attributes_df.index = attributes_df.index.str.replace(' ', '+')
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
            image = Image.new('RGB', (224, 224))
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
    if val_losses:
        plt.plot(epochs[:len(val_losses)], val_losses, 'r-', label='Validation Loss')
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
    if val_jaccards:
        plt.plot(epochs[:len(val_jaccards)], val_jaccards, 'r-', label='Validation Jaccard Accuracy')
    plt.title(f'{model_name} Training and Validation Jaccard Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Jaccard Accuracy')
    plt.legend()
    acc_plot_path = os.path.join(output_dir, acc_plot_filename)
    plt.savefig(acc_plot_path)
    plt.close()
    print(f'Training and validation accuracy plot saved to {acc_plot_path}')

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

    report = classification_report(labels.astype(int), predictions.astype(int), target_names=attribute_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    if trial_number == 'initial':
        report_filename = f'classification_report_{model_name}.csv'
    else:
        report_filename = f'classification_report_{model_name}_trial{trial_number}.csv'
    report_output_path = os.path.join(output_dir, report_filename)
    report_df.to_csv(report_output_path)
    print(f'Classification report saved to {report_output_path}')

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

def load_best_hyperparams():
    if not os.path.exists(performance_summary_path):
        raise FileNotFoundError("No performance summary found. Run round 1 first.")
    df = pd.read_csv(performance_summary_path)
    df_res = df[df['Model'] == 'resnet50']
    if df_res.empty:
        raise ValueError("No entries for resnet50 found in the performance summary.")

    # Select best trial by highest Best Validation Jaccard
    best_row = df_res.loc[df_res['Best Validation Jaccard'].idxmax()]

    # Extract hyperparams
    best_params = {}
    best_params['optimizer'] = best_row.get('Optimizer', 'Adam')
    best_params['learning_rate'] = float(best_row.get('Learning rate', 1e-4))
    best_params['batch_size'] = int(best_row.get('Batch size', 32))
    best_params['weight_decay'] = float(best_row.get('Weight decay', 1e-5))
    best_params['T_0'] = int(best_row.get('T 0', 10))
    best_params['threshold'] = float(best_row.get('Threshold', 0.5))
    best_params['dropout_rate'] = float(best_row.get('Dropout rate', 0.5))
    return best_params

def get_narrowed_search_space(best_params):
    # Just as an example, narrow each param range around the best found value:
    lr_center = best_params['learning_rate']
    lr_low = max(lr_center/2, 1e-6)
    lr_high = min(lr_center*2, 1e-2)

    wd_center = best_params['weight_decay']
    wd_low = max(wd_center/2, 1e-7)
    wd_high = min(wd_center*2, 1e-3)

    thr_center = best_params['threshold']
    thr_low = max(0.3, thr_center - 0.1)
    thr_high = min(0.7, thr_center + 0.1)

    drop_center = best_params['dropout_rate']
    drop_low = max(0.3, drop_center - 0.1)
    drop_high = min(0.7, drop_center + 0.1)

    T0_center = best_params['T_0']
    T0_low = max(10, T0_center - 10)
    T0_high = min(50, T0_center + 10)

    # If best batch_size was 32, let's keep same or just a small set:
    # If best optimizer is known, let's still allow ['Adam', 'SGD']
    return {
        'batch_size': [best_params['batch_size']] if best_params['batch_size'] in [16,32,64] else [32,64],
        'learning_rate': (lr_low, lr_high),
        'weight_decay': (wd_low, wd_high),
        'optimizer': ['Adam', 'SGD'],
        'T_0': (T0_low, T0_high),
        'threshold': (thr_low, thr_high),
        'dropout_rate': (drop_low, drop_high)
    }

def train_with_warmup(params, trial_number='second_round'):
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']
    optimizer_name = params['optimizer']
    T_0 = params['T_0']
    threshold = params['threshold']
    dropout_rate = params['dropout_rate']

    # Long training transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),
        'validate': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
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

    from torchvision.models import resnet50
    base_model = resnet50(weights=None)

    class ResNet50WithDropout(nn.Module):
        def __init__(self, original_model, dropout_rate=0.5):
            super(ResNet50WithDropout, self).__init__()
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.dropout = nn.Dropout(p=dropout_rate)
            self.fc = nn.Linear(original_model.fc.in_features, num_attributes)

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.fc(x)
            return x

    model = ResNet50WithDropout(base_model, dropout_rate=dropout_rate)
    model = model.to(device)

    criterion = SoftJaccardLoss()

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=2)

    # We'll do early stopping only after we start validation
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)

    model_name = 'resnet50_2nd_round'
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

    reached_90 = False

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Always train phase
        model.train()
        running_loss = 0.0
        running_jaccard = 0.0
        progress_bar = tqdm(enumerate(dataloaders['train']), desc=f"Train Epoch {epoch+1}",
                            total=len(dataloaders['train']), unit='batch')

        for batch_idx, (inputs, labels, img_names) in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.sigmoid(outputs)
            preds_binary = (preds >= threshold).float()
            loss.backward()
            optimizer.step()

            preds_np = preds_binary.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
            batch_jaccard = jaccard_score(labels_np, preds_np, average='samples', zero_division=0)
            running_jaccard += batch_jaccard * inputs.size(0)
            running_loss += loss.item() * inputs.size(0)

            batch_loss = running_loss / ((batch_idx + 1) * inputs.size(0))
            batch_jaccard_avg = running_jaccard / ((batch_idx + 1) * inputs.size(0))
            progress_bar.set_postfix({'Loss': f'{batch_loss:.4f}', 'Jaccard': f'{batch_jaccard_avg:.4f}'})

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_jaccard = running_jaccard / dataset_sizes['train']
        train_losses.append(epoch_loss)
        train_jaccards.append(epoch_jaccard)
        scheduler.step()
        print(f'\nTrain Loss: {epoch_loss:.4f} Jaccard: {epoch_jaccard:.4f}')

        # Check if we reached 0.90 training Jaccard
        if not reached_90 and epoch_jaccard >= training_jaccard_threshold:
            reached_90 = True
            print(f"Reached training Jaccard of {training_jaccard_threshold}, starting validation from next epoch.")

        # Once we have reached the threshold, do validation
        if reached_90:
            model.eval()
            val_predictions = []
            val_labels = []
            val_img_names = []
            running_loss_val = 0.0
            running_jaccard_val = 0.0

            progress_bar_val = tqdm(enumerate(dataloaders['validate']), desc=f"Validate Epoch {epoch+1}",
                                    total=len(dataloaders['validate']), unit='batch')
            with torch.no_grad():
                for batch_idx, (inputs, labels, img_names) in progress_bar_val:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.sigmoid(outputs)
                    preds_binary = (preds >= threshold).float()

                    preds_np = preds_binary.detach().cpu().numpy()
                    labels_np = labels.detach().cpu().numpy()
                    batch_jaccard = jaccard_score(labels_np, preds_np, average='samples', zero_division=0)
                    running_jaccard_val += batch_jaccard * inputs.size(0)
                    running_loss_val += loss.item() * inputs.size(0)

                    val_predictions.append(preds_np)
                    val_labels.append(labels_np)
                    val_img_names.extend(img_names)

            val_predictions = np.vstack(val_predictions)
            val_labels = np.vstack(val_labels)
            epoch_loss_val = running_loss_val / dataset_sizes['validate']
            epoch_jaccard_val = running_jaccard_val / dataset_sizes['validate']
            val_losses.append(epoch_loss_val)
            val_jaccards.append(epoch_jaccard_val)

            print(f'\nValidate Loss: {epoch_loss_val:.4f} Jaccard: {epoch_jaccard_val:.4f}')

            early_stopping(epoch_loss_val)
            if epoch_jaccard_val > best_jaccard:
                best_jaccard = epoch_jaccard_val
                best_model_wts = copy.deepcopy(model.state_dict())
                best_val_predictions = val_predictions
                best_val_labels = val_labels
                best_val_img_names = val_img_names

            if early_stopping.early_stop:
                print("Early stopping")
                break

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
    print(f'Best Validation Jaccard Score: {best_jaccard:.4f}')

    model.load_state_dict(best_model_wts)
    model_save_path = os.path.join(output_dir, f'best_model_resnet50_2nd_round_{trial_number}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'Best model saved to {model_save_path}')

    best_val_loss = val_losses[-1] if val_losses else float('inf')
    if best_val_predictions is not None and best_val_labels is not None and best_val_img_names is not None:
        save_predictions(best_val_predictions, best_val_labels, best_val_img_names, trial_number, 'resnet50_2nd_round')
    else:
        print("No validation predictions to save.")

    plot_training_curves(train_losses, val_losses, train_jaccards, val_jaccards, trial_number, 'resnet50_2nd_round')

    save_performance_summary('resnet50_2nd_round', best_jaccard, best_val_loss, time_elapsed, trial_number, params)

    return best_val_loss

def optuna_objective(trial):
    best_params = load_best_hyperparams()
    space = get_narrowed_search_space(best_params)

    batch_size = trial.suggest_categorical('batch_size', space['batch_size'])
    learning_rate = trial.suggest_float('learning_rate', *space['learning_rate'], log=True)
    weight_decay = trial.suggest_float('weight_decay', *space['weight_decay'], log=True)
    optimizer_name = trial.suggest_categorical('optimizer', space['optimizer'])
    T_0 = trial.suggest_int('T_0', *space['T_0'], step=10)
    threshold = trial.suggest_float('threshold', *space['threshold'], step=0.05)
    dropout_rate = trial.suggest_float('dropout_rate', *space['dropout_rate'], step=0.05)

    params = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'optimizer': optimizer_name,
        'T_0': T_0,
        'threshold': threshold,
        'dropout_rate': dropout_rate
    }

    val_loss = train_with_warmup(params, trial_number=trial.number)
    return val_loss

if __name__ == '__main__':
    # Second round search
    study = optuna.create_study(direction='minimize')
    study.optimize(optuna_objective, n_trials=n_trials)

    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    print(f'  Trial Number: {trial.number}')
    print(f'  Loss: {trial.value}')
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

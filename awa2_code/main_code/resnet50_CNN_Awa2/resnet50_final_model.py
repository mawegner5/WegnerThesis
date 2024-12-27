"""
resnet50_final_model.py

This script trains a ResNet50 model with the best hyperparameters found 
in previous rounds. The logic is:

1. Train up to 1500 epochs max.
2. Track training Jaccard each epoch. We do NOT do early stopping until
   the model first achieves >=90% Jaccard on the training set (at least once).
3. Once 90% train Jaccard is reached, we activate early stopping based 
   on validation loss (patience=40).
4. We keep track of validation metrics the entire time for logging, 
   and save the best model by validation Jaccard.
5. We log all epoch-level data (train loss, train jaccard, val loss, 
   val jaccard) to a CSV.

"""

import os
import time
import copy
import csv
import datetime
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import jaccard_score, classification_report
import matplotlib.pyplot as plt

########################################
# User Settings
########################################

# Data / Paths
data_dir = "/remote_home/WegnerThesis/animals_with_attributes/animals_w_att_data"
output_dir = "/remote_home/WegnerThesis/charts_figures_etc"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

performance_summary_path = os.path.join(output_dir, "model_performance_summary.csv")

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Logging
train_log_filename = os.path.join(output_dir, "resnet50_final_training_log.csv")

# Hyperparameters (from best found)
max_epochs = 1500                # Hard cap on epochs
batch_size = 32
learning_rate = 1e-4
weight_decay = 1e-5
optimizer_name = "Adam"
threshold = 0.5
dropout_rate = 0.5
T_0 = 10  # for CosineAnnealingWarmRestarts

# Training strategy
train_jaccard_target = 0.90  # until we get 90% on train
early_stopping_patience = 40  # once train jaccard hits 90, 
                              # we do early stopping on val loss w/ patience=40

########################################
# Data Setup
########################################

attributes_csv_path = os.path.join(data_dir, "predicate_matrix_with_labels.csv")
attributes_df = pd.read_csv(attributes_csv_path, index_col=0)

# Ensure class names match directory format (replace spaces with '+')
attributes_df.index = attributes_df.index.str.replace(" ", "+")
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

        # Build mapping from class -> attribute vector
        self.class_to_attributes = attributes_df.to_dict(orient="index")
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
            class_attributes = np.array(
                list(self.class_to_attributes[class_name].values()), dtype=np.float32
            )
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
            image = Image.new("RGB", (224, 224))
        if self.transform is not None:
            image = self.transform(image)
        attributes = torch.FloatTensor(attributes)
        return image, attributes, img_name

########################################
# Transforms
########################################

data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),
    "validate": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),
}

########################################
# Datasets / Dataloaders
########################################

datasets_dict = {
    "train": AwA2Dataset(data_dir, "train", transform=data_transforms["train"]),
    "validate": AwA2Dataset(data_dir, "validate", transform=data_transforms["validate"]),
}

dataloaders = {
    "train": DataLoader(datasets_dict["train"], batch_size=batch_size,
                        shuffle=True, num_workers=0, pin_memory=False),
    "validate": DataLoader(datasets_dict["validate"], batch_size=batch_size,
                           shuffle=False, num_workers=0, pin_memory=False),
}

dataset_sizes = {x: len(datasets_dict[x]) for x in ["train", "validate"]}

########################################
# Model Definition (ResNet50 + Dropout)
########################################

from torchvision.models import resnet50

class ResNet50WithDropout(nn.Module):
    def __init__(self, original_model, dropout_rate=0.5):
        super(ResNet50WithDropout, self).__init__()
        # Take all layers except the final FC
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(original_model.fc.in_features, num_attributes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

base_model = resnet50(weights=None)
model = ResNet50WithDropout(base_model, dropout_rate=dropout_rate)
model = model.to(device)

########################################
# Loss / Optim / Scheduler
########################################

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

criterion = SoftJaccardLoss()

if optimizer_name.lower() == "adam":
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
else:
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)

scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=2)

########################################
# Custom EarlyStopping that activates after 90% train Jaccard
########################################

class DelayedEarlyStopping:
    """
    This class only activates early stopping based on validation loss 
    *after* the training Jaccard has reached a certain threshold 
    (train_jaccard_target).
    """
    def __init__(self, patience=40, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.active = False  # not active until train_jaccard >= threshold

    def activate(self):
        if not self.active:
            self.active = True
            self.counter = 0
            self.best_loss = None
            if self.verbose:
                print("Early stopping is now ACTIVE (train_jaccard >= target).")

    def __call__(self, val_loss):
        if not self.active:
            # No early stopping if not active
            return
        # If active, do the normal logic
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.verbose:
                print(f'[EarlyStopping] Validation loss set to {val_loss:.6f}')
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'[EarlyStopping] counter: {self.counter}/{self.patience}, no improvement from {self.best_loss:.6f}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("[EarlyStopping] Patience exceeded, early stopping triggered!")
        else:
            if self.verbose:
                print(f'[EarlyStopping] Validation loss improved from {self.best_loss:.6f} to {val_loss:.6f}')
            self.best_loss = val_loss
            self.counter = 0

########################################
# Setup logs
########################################

log_header = ["epoch", "train_loss", "train_jaccard", "val_loss", "val_jaccard"]
if os.path.exists(train_log_filename):
    os.remove(train_log_filename)
with open(train_log_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(log_header)

########################################
# Training Loop
########################################

early_stopper = DelayedEarlyStopping(patience=early_stopping_patience, verbose=True, delta=0)
best_val_jaccard = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
best_val_predictions = None
best_val_labels = None
best_val_img_names = None

train_jaccard_threshold_reached = False

start_time = time.time()

for epoch in range(1, max_epochs + 1):
    print(f"\nEpoch {epoch}/{max_epochs}")
    print("-" * 10)

    #--- Train Phase ---
    model.train()
    running_loss_train = 0.0
    running_jaccard_train = 0.0
    train_loader = dataloaders["train"]
    progress_bar = tqdm(enumerate(train_loader), desc=f"Train Epoch {epoch}", total=len(train_loader), unit="batch")

    for batch_idx, (inputs, labels, img_names) in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        preds = torch.sigmoid(outputs)
        preds_binary = (preds >= threshold).float()

        loss.backward()
        optimizer.step()

        # stats
        preds_np = preds_binary.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        batch_jacc = jaccard_score(labels_np, preds_np, average="samples", zero_division=0)
        running_jaccard_train += batch_jacc * inputs.size(0)
        running_loss_train += loss.item() * inputs.size(0)

        current_loss = running_loss_train / ((batch_idx + 1) * inputs.size(0))
        current_jacc = running_jaccard_train / ((batch_idx + 1) * inputs.size(0))

        progress_bar.set_postfix({"Loss": f"{current_loss:.4f}", "Jaccard": f"{current_jacc:.4f}"})

    epoch_loss_train = running_loss_train / dataset_sizes["train"]
    epoch_jacc_train = running_jaccard_train / dataset_sizes["train"]
    print(f"\nTrain Loss: {epoch_loss_train:.4f} Jaccard: {epoch_jacc_train:.4f}")

    # Step the scheduler every epoch after training
    scheduler.step()

    # Check if we reached the 90% threshold on train
    if not train_jaccard_threshold_reached and epoch_jacc_train >= train_jaccard_target:
        train_jaccard_threshold_reached = True
        early_stopper.activate()  # Now we start applying early stopping

    #--- Validation Phase (Always compute, but early stopping only after threshold) ---
    epoch_loss_val = None
    epoch_jacc_val = None

    val_loader = dataloaders["validate"]
    model.eval()
    val_predictions = []
    val_labels_list = []
    val_img_names_list = []
    running_loss_val = 0.0
    running_jaccard_val = 0.0

    with torch.no_grad():
        progress_bar_val = tqdm(enumerate(val_loader), desc=f"Val Epoch {epoch}", total=len(val_loader), unit="batch")
        for batch_idx, (inputs, labels, img_names) in progress_bar_val:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss_val = criterion(outputs, labels)
            preds_val = torch.sigmoid(outputs)
            preds_binary_val = (preds_val >= threshold).float()

            preds_np_val = preds_binary_val.detach().cpu().numpy()
            labels_np_val = labels.detach().cpu().numpy()
            batch_jacc_val = jaccard_score(labels_np_val, preds_np_val, average="samples", zero_division=0)

            running_jaccard_val += batch_jacc_val * inputs.size(0)
            running_loss_val += loss_val.item() * inputs.size(0)

            val_predictions.append(preds_np_val)
            val_labels_list.append(labels_np_val)
            val_img_names_list.extend(img_names)

            current_loss_val = running_loss_val / ((batch_idx + 1) * inputs.size(0))
            current_jacc_val = running_jaccard_val / ((batch_idx + 1) * inputs.size(0))
            progress_bar_val.set_postfix({"Loss": f"{current_loss_val:.4f}", "Jaccard": f"{current_jacc_val:.4f}"})

    epoch_loss_val = running_loss_val / dataset_sizes["validate"]
    epoch_jacc_val = running_jaccard_val / dataset_sizes["validate"]
    print(f"\nVal Loss: {epoch_loss_val:.4f} Jaccard: {epoch_jacc_val:.4f}")

    #--- Check if we have a new best Val Jaccard ---
    if epoch_jacc_val > best_val_jaccard:
        best_val_jaccard = epoch_jacc_val
        best_model_wts = copy.deepcopy(model.state_dict())

        # Save predictions
        best_val_predictions = np.vstack(val_predictions)
        best_val_labels = np.vstack(val_labels_list)
        best_val_img_names = val_img_names_list

        # Save model checkpoint
        best_model_path = os.path.join(output_dir, "best_model_resnet50_final.pth")
        torch.save(best_model_wts, best_model_path)
        print(f"New best val Jaccard: {best_val_jaccard:.4f} -> model saved to {best_model_path}")

    #--- Early Stopping (only if train_jaccard_threshold_reached) ---
    early_stopper(epoch_loss_val)
    if early_stopper.early_stop:
        print("Early stopping triggered after 90% train jaccard was reached + patience exhausted on val loss.")
        break

    #--- Log to CSV ---
    with open(train_log_filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{epoch_loss_train:.6f}",
            f"{epoch_jacc_train:.6f}",
            f"{epoch_loss_val:.6f}",
            f"{epoch_jacc_val:.6f}"
        ])

    #--- If we've reached max_epochs, we stop anyway ---
    if epoch == max_epochs:
        print(f"Reached max epochs ({max_epochs}). Training done.")

train_time = time.time() - start_time
print(f"\nTraining complete in {int(train_time // 60)}m {int(train_time % 60)}s")
print(f"Best Validation Jaccard Score: {best_val_jaccard:.4f}")

#--- If we have best val predictions, save them ---
def save_val_predictions_final():
    if best_val_predictions is not None and best_val_labels is not None and best_val_img_names is not None:
        filename = "predictions_resnet50_final.csv"
        output_path = os.path.join(output_dir, filename)
        df_predictions = pd.DataFrame(best_val_predictions.astype(int), columns=attribute_names)
        df_predictions.insert(0, "image_name", best_val_img_names)
        df_predictions.to_csv(output_path, index=False)
        print(f"Final best validation predictions saved to {output_path}")

        # classification report
        report = classification_report(best_val_labels.astype(int),
                                       best_val_predictions.astype(int),
                                       target_names=attribute_names,
                                       output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        report_filename = "classification_report_resnet50_final.csv"
        report_output_path = os.path.join(output_dir, report_filename)
        report_df.to_csv(report_output_path)
        print(f"Final best validation classification report saved to {report_output_path}")
    else:
        print("No final validation predictions to save.")

save_val_predictions_final()

#--- Save final logs / performance summary
def save_performance_summary_final(model_name, best_jaccard, best_val_loss, train_time):
    data = {
        "Trial": ["final_run"],
        "Model": [model_name],
        "Best Validation Jaccard": [best_jaccard],
        "Best Validation Loss": [best_val_loss],
        "Training Time (s)": [int(train_time)],
        "Timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Optimizer": [optimizer_name],
        "Learning Rate": [learning_rate],
        "Batch Size": [batch_size],
        "Weight Decay": [weight_decay],
        "Num Epochs": [max_epochs],
        "Dropout Rate": [dropout_rate],
        "T_0": [T_0],
        "Threshold": [threshold],
        "Early Stopping Patience": [early_stopping_patience]
    }
    df = pd.DataFrame(data)
    if os.path.exists(performance_summary_path):
        df_existing = pd.read_csv(performance_summary_path)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(performance_summary_path, index=False)
    print(f"Final run performance summary updated at {performance_summary_path}")

# We define best_val_loss as the last validation loss that gave best_jaccard
# That is (conceptually) the time we found best_jaccard. We'll approximate
# with the final epoch's val loss or the best val loss recorded at that time.
final_val_loss = val_losses[-1] if val_losses else float("inf")
save_performance_summary_final("resnet50_final", best_val_jaccard, final_val_loss, train_time)

print("Done. Ready to test on the final model at a later time.")

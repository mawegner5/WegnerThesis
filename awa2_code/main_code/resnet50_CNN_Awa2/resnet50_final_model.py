#!/usr/bin/env python3
"""
resnet50_final_model.py

This script trains a ResNet50 model with a set of fixed hyperparameters
until the training Jaccard reaches 95% at least once, after which
early stopping is allowed to trigger based on validation loss (patience=40).

Additionally:
 - Saves all outputs (model weights, logs, CSVs) to /remote_home/WegnerThesis/test_outputs/final_resnet50_training_output
 - Plots training/validation loss and Jaccard curves
 - Plots a per-attribute F1 bar chart from the final classification report
   to give more insight for your thesis.

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
# 0. Where to save everything
########################################

output_dir = "/remote_home/WegnerThesis/test_outputs/final_resnet50_training_output"
os.makedirs(output_dir, exist_ok=True)
performance_summary_path = os.path.join(output_dir, "model_performance_summary.csv")

########################################
# 1. Hyperparameters / Config
########################################

# Data / Paths
data_dir = "/remote_home/WegnerThesis/animals_with_attributes/animals_w_att_data"

# Training
max_epochs = 1500
batch_size = 32
learning_rate = 1e-4
weight_decay = 1e-5
optimizer_name = "Adam"
threshold = 0.5
dropout_rate = 0.5
T_0 = 10  # for CosineAnnealingWarmRestarts
train_jaccard_target = 0.95
early_stopping_patience = 40

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

########################################
# Logging
########################################
train_log_filename = os.path.join(output_dir, "resnet50_final_training_log.csv")

########################################
# Dataset Setup
########################################

attributes_csv_path = os.path.join(data_dir, "predicate_matrix_with_labels.csv")
attributes_df = pd.read_csv(attributes_csv_path, index_col=0)

# Replace spaces with '+' in class names to match directory structure
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
        self.class_to_attributes = attributes_df.to_dict(orient="index")
        self._prepare_dataset()

    def _prepare_dataset(self):
        phase_dir = os.path.join(self.root_dir, self.phase)
        for class_name in os.listdir(phase_dir):
            class_dir = os.path.join(phase_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            if class_name not in self.class_to_attributes:
                print(f"Warning: Class {class_name} not in attribute list.")
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
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    "validate": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

########################################
# DataLoaders
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

print(f"Dataset sizes: train={dataset_sizes['train']}, validate={dataset_sizes['validate']}")

########################################
# Model (ResNet50 + Dropout)
########################################

from torchvision.models import resnet50


class ResNet50WithDropout(nn.Module):
    def __init__(self, original_model, dropout_rate=0.5):
        super().__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])  # all but final FC
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
        super().__init__()
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
# Delayed Early Stopping
########################################

class DelayedEarlyStopping:
    def __init__(self, patience=40, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta
        self.active = False

    def activate(self):
        if not self.active:
            self.active = True
            self.counter = 0
            self.best_loss = None
            if self.verbose:
                print("Early stopping is now ACTIVE (train_jaccard >= threshold).")

    def __call__(self, val_loss):
        if not self.active:
            return
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
# CSV Log Setup
########################################

if os.path.exists(train_log_filename):
    os.remove(train_log_filename)

with open(train_log_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_jaccard", "val_loss", "val_jaccard"])

########################################
# Training Loop
########################################

early_stopper = DelayedEarlyStopping(patience=early_stopping_patience, verbose=True, delta=0)
train_jaccard_threshold_reached = False

best_val_jaccard = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

best_val_predictions = None
best_val_labels = None
best_val_img_names = None

train_losses = []
val_losses = []
train_jaccs = []
val_jaccs = []

start_time = time.time()

for epoch in range(1, max_epochs + 1):
    print(f"\nEpoch {epoch}/{max_epochs}")
    print("-" * 10)

    # --- TRAIN PHASE ---
    model.train()
    running_loss_train = 0.0
    running_jaccard_train = 0.0

    train_loader = dataloaders["train"]
    progress_bar = tqdm(enumerate(train_loader), desc=f"Train Epoch {epoch}",
                        total=len(train_loader), unit="batch")

    for batch_idx, (inputs, labels, img_names) in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        preds = torch.sigmoid(outputs)
        preds_binary = (preds >= threshold).float()

        loss.backward()
        optimizer.step()

        # accumulate stats
        batch_size_actual = inputs.size(0)
        running_loss_train += loss.item() * batch_size_actual

        preds_np = preds_binary.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        batch_jacc = jaccard_score(labels_np, preds_np, average="samples", zero_division=0)
        running_jaccard_train += batch_jacc * batch_size_actual

        # show progress
        current_loss = running_loss_train / ((batch_idx + 1) * batch_size_actual)
        current_jacc = running_jaccard_train / ((batch_idx + 1) * batch_size_actual)
        progress_bar.set_postfix({"Loss": f"{current_loss:.4f}", 
                                  "Jaccard": f"{current_jacc:.4f}"})

    epoch_loss_train = running_loss_train / dataset_sizes["train"]
    epoch_jacc_train = running_jaccard_train / dataset_sizes["train"]
    train_losses.append(epoch_loss_train)
    train_jaccs.append(epoch_jacc_train)

    print(f"Train Loss: {epoch_loss_train:.4f}, Train Jaccard: {epoch_jacc_train:.4f}")

    scheduler.step()  # step LR schedule each epoch

    # check if we reached 95% train jacc
    if not train_jaccard_threshold_reached and epoch_jacc_train >= train_jaccard_target:
        train_jaccard_threshold_reached = True
        early_stopper.activate()

    # --- VALIDATION PHASE ---
    model.eval()
    val_loader = dataloaders["validate"]
    running_loss_val = 0.0
    running_jaccard_val = 0.0

    val_predictions = []
    val_labels_list = []
    val_img_names_list = []

    with torch.no_grad():
        progress_bar_val = tqdm(enumerate(val_loader), desc=f"Val Epoch {epoch}",
                                total=len(val_loader), unit="batch")
        for batch_idx, (inputs, labels, img_names) in progress_bar_val:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss_val = criterion(outputs, labels)

            preds_val = torch.sigmoid(outputs)
            preds_binary_val = (preds_val >= threshold).float()

            batch_size_val = inputs.size(0)
            running_loss_val += loss_val.item() * batch_size_val

            preds_np_val = preds_binary_val.detach().cpu().numpy()
            labels_np_val = labels.detach().cpu().numpy()
            batch_jacc_val = jaccard_score(labels_np_val, preds_np_val, average="samples", zero_division=0)
            running_jaccard_val += batch_jacc_val * batch_size_val

            val_predictions.append(preds_np_val)
            val_labels_list.append(labels_np_val)
            val_img_names_list.extend(img_names)

            # show progress
            current_loss_val = running_loss_val / ((batch_idx + 1) * batch_size_val)
            current_jacc_val = running_jaccard_val / ((batch_idx + 1) * batch_size_val)
            progress_bar_val.set_postfix({"Loss": f"{current_loss_val:.4f}",
                                          "Jaccard": f"{current_jacc_val:.4f}"})

    epoch_loss_val = running_loss_val / dataset_sizes["validate"]
    epoch_jacc_val = running_jaccard_val / dataset_sizes["validate"]
    val_losses.append(epoch_loss_val)
    val_jaccs.append(epoch_jacc_val)

    print(f"Val Loss: {epoch_loss_val:.4f}, Val Jaccard: {epoch_jacc_val:.4f}")

    # check if best val jacc
    if epoch_jacc_val > best_val_jaccard:
        best_val_jaccard = epoch_jacc_val
        best_model_wts = copy.deepcopy(model.state_dict())

        best_val_predictions = np.vstack(val_predictions)
        best_val_labels = np.vstack(val_labels_list)
        best_val_img_names = val_img_names_list

        best_model_path = os.path.join(output_dir, "best_model_resnet50_final.pth")
        torch.save(best_model_wts, best_model_path)
        print(f"[INFO] New best val Jaccard: {best_val_jaccard:.4f} -> saved to {best_model_path}")

    # early stopping
    early_stopper(epoch_loss_val)
    if early_stopper.early_stop:
        print("[INFO] Early stopping triggered after 95% train jacc + patience on val loss.")
        break

    # Write CSV log
    with open(train_log_filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{epoch_loss_train:.6f}",
            f"{epoch_jacc_train:.6f}",
            f"{epoch_loss_val:.6f}",
            f"{epoch_jacc_val:.6f}"
        ])

    if epoch == max_epochs:
        print(f"[INFO] Reached max_epochs ({max_epochs}). End training.")
        break

total_train_time = time.time() - start_time
print(f"[INFO] Training complete in {int(total_train_time // 60)}m {int(total_train_time % 60)}s")
print(f"[INFO] Best Validation Jaccard Score: {best_val_jaccard:.4f}")

########################################
# Plot Training/Validation Curves
########################################

# 1) Plot Loss
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses,   label="Val Loss")
plt.title("ResNet50 Final: Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
loss_plot_path = os.path.join(output_dir, "resnet50_final_training_validation_loss.png")
plt.savefig(loss_plot_path)
plt.close()
print(f"[INFO] Saved training vs validation loss plot: {loss_plot_path}")

# 2) Plot Jaccard
plt.figure()
plt.plot(train_jaccs, label="Train Jaccard")
plt.plot(val_jaccs,   label="Val Jaccard")
plt.title("ResNet50 Final: Training vs Validation Jaccard Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Jaccard")
plt.legend()
jacc_plot_path = os.path.join(output_dir, "resnet50_final_training_validation_jaccard.png")
plt.savefig(jacc_plot_path)
plt.close()
print(f"[INFO] Saved training vs validation Jaccard plot: {jacc_plot_path}")

########################################
# Save final predictions & classification report
########################################

if best_val_predictions is not None and best_val_labels is not None and best_val_img_names is not None:
    # Save predictions
    val_pred_csv = os.path.join(output_dir, "predictions_resnet50_final.csv")
    df_predictions = pd.DataFrame(best_val_predictions.astype(int), columns=attribute_names)
    df_predictions.insert(0, "image_name", best_val_img_names)
    df_predictions.to_csv(val_pred_csv, index=False)
    print(f"[INFO] Saved best validation predictions to {val_pred_csv}")

    # Classification report
    report = classification_report(best_val_labels.astype(int),
                                   best_val_predictions.astype(int),
                                   target_names=attribute_names,
                                   output_dict=True,
                                   zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_filename = "classification_report_resnet50_final.csv"
    report_output_path = os.path.join(output_dir, report_filename)
    report_df.to_csv(report_output_path)
    print(f"[INFO] Saved final validation classification report -> {report_output_path}")

    # 3) Optional: Per-attribute F1 bar chart
    #    We'll parse the "f1-score" from each attribute row
    #    The classification_report has one row per attribute name
    #    and "macro avg", "weighted avg", etc.

    # We only want the attributes themselves (not the averages).
    # attribute_names appear in the index, let's filter out typical summary keys:
    exclude_keys = {"accuracy", "macro avg", "weighted avg", "micro avg"}
    f1_scores = []
    attr_labels = []
    for idx in report_df.index:
        if idx not in exclude_keys:
            # idx is the attribute name
            # the "f1-score" column has the numeric value
            f1_val = report_df.loc[idx, "f1-score"]
            f1_scores.append(f1_val)
            attr_labels.append(idx)

    # Let's do a bar chart with the top N or all attributes
    # If we have 85 attributes, it might be large. We can show them all or choose 20 worst/best.
    # Here, we do all but keep the plot big:
    plt.figure(figsize=(20, 8))
    x_pos = np.arange(len(attr_labels))
    plt.bar(x_pos, f1_scores, color="blue")
    plt.xticks(x_pos, attr_labels, rotation=90)
    plt.ylim([0, 1])
    plt.title("Per-Attribute F1 (Validation)")
    plt.xlabel("Attribute")
    plt.ylabel("F1 Score")

    bar_chart_path = os.path.join(output_dir, "resnet50_final_per_attribute_f1.png")
    plt.tight_layout()
    plt.savefig(bar_chart_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved per-attribute F1 bar chart: {bar_chart_path}")
else:
    print("[WARN] No final validation predictions to save or analyze for attribute metrics.")

########################################
# Save performance summary CSV (model_performance_summary.csv)
########################################

def save_performance_summary(model_name, best_val_jacc, val_losses_list, train_time):
    best_val_loss = float(np.min(val_losses_list)) if len(val_losses_list)>0 else float("inf")

    data = {
        "Trial": ["final_run"],
        "Model": [model_name],
        "Best Validation Jaccard": [best_val_jacc],
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
        "Early Stopping Patience": [early_stopping_patience],
        "Train Jaccard Target": [train_jaccard_target]
    }
    df = pd.DataFrame(data)
    if os.path.exists(performance_summary_path):
        df_existing = pd.read_csv(performance_summary_path)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(performance_summary_path, index=False)
    print(f"[INFO] Performance summary updated at {performance_summary_path}")

save_performance_summary("resnet50_final", best_val_jaccard, val_losses, total_train_time)

print("[DONE] ResNet50 final training script complete. All outputs saved.")

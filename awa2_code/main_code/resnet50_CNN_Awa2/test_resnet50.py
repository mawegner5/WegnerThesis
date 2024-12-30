########################################################
# test_resnet50.py
#
# This script loads a trained ResNet50 model (with dropout),
# evaluates it on the chosen dataset (validate or test),
# and produces CSV outputs:
#   1) Per-image CSV: [image_name, actual_species, ground_truth_attributes, predicted_attributes]
#   2) Per-attribute confusion info: [attribute_name, TP, FP, TN, FN, precision, recall, F1]
#
# The outputs go to /remote_home/WegnerThesis/test_outputs
#
# Adjust as needed for your environment.
########################################################

import os
import csv
import time
import copy
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import jaccard_score, classification_report, confusion_matrix

########################################
# User Settings
########################################

# Where the best model is saved (from your final training)
BEST_MODEL_PATH = "/remote_home/WegnerThesis/charts_figures_etc/best_model_resnet50_final.pth"

# Data / Paths
DATA_DIR = "/remote_home/WegnerThesis/animals_with_attributes/animals_w_att_data"
OUTPUT_DIR = "/remote_home/WegnerThesis/test_outputs"  # <-- main output directory
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Choose which dataset to evaluate: "validate" or "test"
PHASE_TO_EVAL = "test"  # or "validate"

# Model hyperparams used in training
DROPOUT_RATE = 0.5
THRESHOLD = 0.5  # For predicted attribute

# Dataloader
BATCH_SIZE = 32
NUM_WORKERS = 0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

########################################
# Load Attribute Matrix
########################################

ATTR_MATRIX_CSV = os.path.join(DATA_DIR, "predicate_matrix_with_labels.csv")
attributes_df = pd.read_csv(ATTR_MATRIX_CSV, index_col=0)
# Convert any spaces in class names to '+'
attributes_df.index = attributes_df.index.str.replace(" ", "+")
attribute_names = attributes_df.columns.tolist()
classes = attributes_df.index.tolist()
num_attributes = len(attribute_names)

########################################
# Custom Dataset for AWA2
########################################

class AwA2Dataset(Dataset):
    def __init__(self, root_dir, phase, transform=None):
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform
        self.samples = []
        self.attributes = []
        # Build mapping from class->attribute vector
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
                self.samples.append((img_path, img_name, class_name))  # store actual species
                self.attributes.append(class_attributes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, img_name, class_name = self.samples[idx]
        attributes = self.attributes[idx]
        try:
            image = datasets.folder.default_loader(img_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # fallback blank image
            image = Image.new("RGB", (224,224))
        if self.transform is not None:
            image = self.transform(image)
        attributes_tensor = torch.FloatTensor(attributes)
        return image, attributes_tensor, img_name, class_name

########################################
# Data Transforms
########################################

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

########################################
# Create Dataset / Dataloader
########################################

eval_dataset = AwA2Dataset(DATA_DIR, PHASE_TO_EVAL, transform=data_transforms)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=False)

########################################
# Model Definition
########################################

from torchvision.models import resnet50

class ResNet50WithDropout(nn.Module):
    def __init__(self, original_model, dropout_rate=0.5):
        super().__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(original_model.fc.in_features, num_attributes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

########################################
# Load the best model
########################################

def load_best_model(model_path):
    base_model = resnet50(weights=None)
    model = ResNet50WithDropout(base_model, dropout_rate=DROPOUT_RATE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_best_model(BEST_MODEL_PATH)
print(f"Loaded best model from {BEST_MODEL_PATH}")

########################################
# Evaluate
########################################

def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))

all_results = []   # For CSV with per-image info
# We'll store [img_name, actual_species, actual_attr_vector, predicted_attr_vector]

# For confusion matrix, we want to accumulate predictions & ground truth per attribute
all_preds_binary = []
all_targets = []

with torch.no_grad():
    for batch_idx, (inputs, targets, img_names, class_names) in enumerate(eval_loader):
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        outputs = model(inputs)          # shape: [batch_size, num_attributes]
        # Convert to CPU for post-processing
        outputs_np = outputs.cpu().numpy()
        targets_np = targets.cpu().numpy()

        # We'll do thresholding
        preds_sigmoid = sigmoid_np(outputs_np)
        preds_binary = (preds_sigmoid >= THRESHOLD).astype(int)

        for i in range(inputs.size(0)):
            # gather info
            actual_sp = class_names[i]  # string
            actual_attr = targets_np[i] # shape [num_attributes]
            pred_attr = preds_binary[i] # shape [num_attributes]
            img_nm = img_names[i]

            all_results.append([
                img_nm,
                actual_sp,
                actual_attr.tolist(),
                pred_attr.tolist()
            ])

        all_preds_binary.append(preds_binary)
        all_targets.append(targets_np)

all_preds_binary = np.concatenate(all_preds_binary, axis=0)  # shape [N, num_attributes]
all_targets = np.concatenate(all_targets, axis=0)           # shape [N, num_attributes]

# we can also compute a multi-label jaccard on the entire dataset
dataset_jaccard = jaccard_score(all_targets, all_preds_binary, average="samples", zero_division=0)
print(f"{PHASE_TO_EVAL} dataset Jaccard Score (average='samples'): {dataset_jaccard:.4f}")

########################################
# Save per-image CSV
########################################

output_csv_path = os.path.join(OUTPUT_DIR, f"resnet50_{PHASE_TO_EVAL}_predictions.csv")
with open(output_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    # headers
    writer.writerow(["image_name", "actual_species", "actual_attributes", "predicted_attributes"])
    for row in all_results:
        writer.writerow(row)

print(f"Saved per-image predictions to {output_csv_path}")

########################################
# Confusion Matrix per attribute
########################################

# We'll produce a table [attribute_name, TP, FP, TN, FN, precision, recall, f1]
# For each attribute k, we gather the entire column of predictions vs. targets

def safe_div(a, b):
    return a / b if b != 0 else 0

confusion_data = []
for k, attr_name in enumerate(attribute_names):
    y_true = all_targets[:, k]
    y_pred = all_preds_binary[:, k]
    # confusion_matrix in sklearn can do 2x2 if we label [0,1], but let's get that directly
    # We want: TP = sum(y_true==1 & y_pred==1), etc.
    # or we can do
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    # cm is 2x2: cm[0,0] = TN, cm[0,1] = FP, cm[1,0] = FN, cm[1,1] = TP
    tn, fp, fn, tp = cm.ravel()

    prec = safe_div(tp, (tp+fp))
    rec = safe_div(tp, (tp+fn))
    f1 = safe_div(2*prec*rec, (prec+rec)) if (prec+rec)>0 else 0

    confusion_data.append({
        "attribute": attr_name,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "precision": prec,
        "recall": rec,
        "f1": f1
    })

confusion_df = pd.DataFrame(confusion_data)
confusion_csv_path = os.path.join(OUTPUT_DIR, f"resnet50_{PHASE_TO_EVAL}_attribute_confusion.csv")
confusion_df.to_csv(confusion_csv_path, index=False)
print(f"Saved attribute confusion metrics to {confusion_csv_path}")

########################################
# Possibly we compute classification_report for the entire multi-label set
# (though classification_report might require one line per attribute)
########################################

report = classification_report(all_targets, all_preds_binary, target_names=attribute_names,
                               output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()
report_csv_path = os.path.join(OUTPUT_DIR, f"resnet50_{PHASE_TO_EVAL}_classification_report.csv")
report_df.to_csv(report_csv_path)
print(f"Saved classification report to {report_csv_path}")

print("Done. Evaluate code complete!")

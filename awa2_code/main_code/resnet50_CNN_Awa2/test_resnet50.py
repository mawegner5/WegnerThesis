########################################################
# test_resnet50.py
#
# Revised version: each attribute is stored in its own CSV column, 
# both actual and predicted. The final CSV columns are:
#   [image_name, actual_species, actual_<attr1>, ..., actual_<attrN>,
#    pred_<attr1>, ..., pred_<attrN>]
#
# This eliminates the "invalid literal for int()" issue in AWA2_LLM.py.
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

BEST_MODEL_PATH = "/remote_home/WegnerThesis/charts_figures_etc/best_model_resnet50_final.pth"

DATA_DIR = "/remote_home/WegnerThesis/animals_with_attributes/animals_w_att_data"
OUTPUT_DIR = "/remote_home/WegnerThesis/test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Evaluate on "validate" or "test":
PHASE_TO_EVAL = "test"

DROPOUT_RATE = 0.5
THRESHOLD = 0.5

BATCH_SIZE = 32
NUM_WORKERS = 0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

########################################
# Load Attribute Matrix
########################################

ATTR_MATRIX_CSV = os.path.join(DATA_DIR, "predicate_matrix_with_labels.csv")
attributes_df = pd.read_csv(ATTR_MATRIX_CSV, index_col=0)
attributes_df.index = attributes_df.index.str.replace(" ", "+")
attribute_names = attributes_df.columns.tolist()
num_attributes = len(attribute_names)

########################################
# Dataset
########################################

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
            class_attrs = np.array(list(self.class_to_attributes[class_name].values()), dtype=np.float32)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, img_name, class_name))
                self.attributes.append(class_attrs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, img_name, class_name = self.samples[idx]
        attr_vec = self.attributes[idx]
        try:
            image = datasets.folder.default_loader(img_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new("RGB", (224,224))
        if self.transform is not None:
            image = self.transform(image)
        attr_tensor = torch.FloatTensor(attr_vec)
        return image, attr_tensor, img_name, class_name

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

eval_dataset = AwA2Dataset(DATA_DIR, PHASE_TO_EVAL, transform=data_transforms)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=False)

########################################
# Model
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

def load_best_model(model_path):
    base_model = resnet50(weights=None)
    model = ResNet50WithDropout(base_model, dropout_rate=DROPOUT_RATE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_best_model(BEST_MODEL_PATH)
print(f"Loaded best model from {BEST_MODEL_PATH}")

def sigmoid_np(x):
    return 1 / (1 + np.exp(-x))

########################################
# Inference & Data Collection
########################################

all_preds_binary = []
all_targets = []
results_records = []  # to store per-image data

with torch.no_grad():
    for (inputs, targets, img_names, class_names) in eval_loader:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)

        outputs = model(inputs)  # shape [batch, num_attributes]
        outputs_np = outputs.cpu().numpy()
        targets_np = targets.cpu().numpy()

        preds_sigmoid = sigmoid_np(outputs_np)
        preds_binary = (preds_sigmoid >= THRESHOLD).astype(int)

        # Accumulate for confusion metrics
        all_preds_binary.append(preds_binary)
        all_targets.append(targets_np)

        # Gather rows for each image
        batch_size_now = inputs.size(0)
        for i in range(batch_size_now):
            image_name = img_names[i]
            actual_species = class_names[i]
            actual_attr_list = targets_np[i]   # shape [num_attributes]
            pred_attr_list   = preds_binary[i] # shape [num_attributes]

            results_records.append((image_name, actual_species,
                                    actual_attr_list, pred_attr_list))

all_preds_binary = np.concatenate(all_preds_binary, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

dataset_jacc = jaccard_score(all_targets, all_preds_binary, average="samples", zero_division=0)
print(f"{PHASE_TO_EVAL} dataset Jaccard Score (average='samples'): {dataset_jacc:.4f}")

########################################
# Save per-image CSV
########################################

out_csv_path = os.path.join(OUTPUT_DIR, f"resnet50_{PHASE_TO_EVAL}_predictions.csv")

# Build the header with user-friendly names
header = (["image_name","actual_species"]
          + [f"Actual_{a}" for a in attribute_names]   # changed label
          + [f"Predicted_{a}" for a in attribute_names])  # changed label

with open(out_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for rec in results_records:
        img_nm, sp, actual_vec, pred_vec = rec
        row = [img_nm, sp]
        # "Actual_" columns
        row.extend([int(x) for x in actual_vec])
        # "Predicted_" columns
        row.extend([int(x) for x in pred_vec])
        writer.writerow(row)

print(f"Saved per-image predictions to {out_csv_path}")


########################################
# Confusion per-attribute
########################################

def safe_div(a, b):
    return a/b if b else 0

confusion_data = []
for i, attr_name in enumerate(attribute_names):
    y_true = all_targets[:, i]
    y_pred = all_preds_binary[:, i]
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()

    prec = safe_div(tp, (tp + fp))
    rec  = safe_div(tp, (tp + fn))
    f1   = safe_div(2*prec*rec, (prec+rec)) if (prec+rec)>0 else 0

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
# Classification report
########################################

report = classification_report(all_targets, all_preds_binary,
                               target_names=attribute_names,
                               output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()
report_csv_path = os.path.join(OUTPUT_DIR, f"resnet50_{PHASE_TO_EVAL}_classification_report.csv")
report_df.to_csv(report_csv_path)
print(f"Saved classification report to {report_csv_path}")

print("Done. Evaluate code complete!")

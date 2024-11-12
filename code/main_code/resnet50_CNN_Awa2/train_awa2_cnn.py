import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.metrics import jaccard_score
from tqdm import tqdm
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# ----------------------------
# Paths and Hyperparameters
# ----------------------------
# Paths
data_dir = '/root/.ipython/WegnerThesis/animals_with_attributes/animals_w_att_data/'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validate')
test_dir = os.path.join(data_dir, 'test')
train_classes_txt = os.path.join(data_dir, 'Animals_with_Attributes2', 'trainclasses.txt')
test_classes_txt = os.path.join(data_dir, 'Animals_with_Attributes2', 'testclasses.txt')
predicates_txt_path = os.path.join(data_dir, 'Animals_with_Attributes2', 'predicates.txt')
attributes_path = os.path.join(data_dir, 'Animals_with_Attributes2', 'predicate-matrix-binary.txt')
output_dir = '/root/.ipython/WegnerThesis/charts_figures_etc/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Hyperparameters
num_epochs = 150  # Adjust as needed
num_attributes = 85  # Number of attributes/predicates in AWA2
patience = 10  # For early stopping
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------
# Custom Dataset Class
# ----------------------------
class AWA2Dataset(Dataset):
    def __init__(self, root_dirs, predicates_txt_path, attributes_path, transform=None):
        self.transform = transform

        # Load attribute names
        self.attributes = []
        with open(predicates_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    attr_name = parts[1]
                    self.attributes.append(attr_name)

        # Load attribute matrix
        self.attribute_matrix = np.loadtxt(attributes_path, dtype=int)  # Shape: (N_classes, N_attributes)

        # Collect all image paths and labels
        self.image_paths = []
        self.labels = []
        for root_dir in root_dirs:
            class_names = os.listdir(root_dir)
            for cls_name in class_names:
                cls_dir = os.path.join(root_dir, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                # Get class index from class name
                class_idx = self.get_class_index(cls_name)
                label = self.attribute_matrix[class_idx]
                for img_name in os.listdir(cls_dir):
                    img_path = os.path.join(cls_dir, img_name)
                    if os.path.isfile(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(label)

    def get_class_index(self, class_name):
        # Class names in attribute matrix are sorted alphabetically
        class_names_sorted = sorted(os.listdir(os.path.join(data_dir, 'Animals_with_Attributes2', 'JPEGImages')))
        class_idx = class_names_sorted.index(class_name)
        return class_idx

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Skip corrupted images
            image = Image.new('RGB', (224, 224))
        label = self.labels[idx]
        label = torch.from_numpy(label).float()
        if self.transform:
            image = self.transform(image)
        return image, label

# ----------------------------
# Data Preparation
# ----------------------------
# Read class names from the provided files
def read_classes(file_path):
    with open(file_path, 'r') as f:
        classes = [line.strip().replace('+', ' ') for line in f.readlines()]
    return classes

# Training and test classes
train_classes_full = read_classes(train_classes_txt)
test_classes = read_classes(test_classes_txt)

# From the training classes, select 5 classes for validation
import random
random.seed(42)  # For reproducibility
random.shuffle(train_classes_full)

val_classes = train_classes_full[:5]     # First 5 classes for validation
train_classes = train_classes_full[5:]   # Remaining 35 classes for training

print(f"Total training classes: {len(train_classes_full)}")
print(f"Training classes ({len(train_classes)}): {train_classes}")
print(f"Validation classes ({len(val_classes)}): {val_classes}")
print(f"Test classes ({len(test_classes)}): {test_classes}")

# Define transforms
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = AWA2Dataset(root_dirs=[os.path.join(train_dir, cls) for cls in train_classes],
                            predicates_txt_path=predicates_txt_path,
                            attributes_path=attributes_path,
                            transform=data_transform)

val_dataset = AWA2Dataset(root_dirs=[os.path.join(val_dir, cls) for cls in val_classes],
                          predicates_txt_path=predicates_txt_path,
                          attributes_path=attributes_path,
                          transform=data_transform)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# ----------------------------
# Hyperparameter Tuning with Ray Tune
# ----------------------------
def train_cnn(config, checkpoint_dir=None):
    # Initialize model
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_attributes)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(config["step_size"]), gamma=config["gamma"])

    # Load checkpoint if available
    if checkpoint_dir:
        checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch_start = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        trigger_times = checkpoint["trigger_times"]
    else:
        epoch_start = 1
        best_val_loss = float('inf')
        trigger_times = 0

    for epoch in range(epoch_start, num_epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_all_targets = []
        val_all_outputs = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)

                outputs = torch.sigmoid(outputs).cpu()
                labels = labels.cpu()
                val_all_outputs.extend(outputs)
                val_all_targets.extend(labels)

        val_epoch_loss = val_running_loss / len(val_dataset)

        # Calculate Jaccard Score
        val_all_outputs = torch.stack(val_all_outputs)
        val_all_targets = torch.stack(val_all_targets)
        val_binarized_outputs = (val_all_outputs > 0.5).float()
        val_jaccard = jaccard_score(val_all_targets.numpy(), val_binarized_outputs.numpy(), average='samples')

        # Update scheduler
        scheduler.step()

        # Early Stopping
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            trigger_times = 0
            # Save checkpoint
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "trigger_times": trigger_times
                }, path)
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break

        # Send metrics to Tune
        tune.report(val_loss=val_epoch_loss, val_jaccard=val_jaccard)

# Hyperparameter search space
config = {
    "lr": tune.loguniform(1e-5, 1e-2),
    "batch_size": tune.choice([16, 32, 64]),
    "weight_decay": tune.loguniform(1e-6, 1e-2),
    "step_size": tune.choice([10, 20, 30]),
    "gamma": tune.uniform(0.1, 0.5)
}

# Scheduler and Reporter for Ray Tune
scheduler = ASHAScheduler(
    metric="val_loss",
    mode="min",
    max_t=num_epochs,
    grace_period=10,
    reduction_factor=2
)

reporter = CLIReporter(
    metric_columns=["val_loss", "val_jaccard", "training_iteration"]
)

# ----------------------------
# Run Hyperparameter Tuning
# ----------------------------
ray.init()

result = tune.run(
    train_cnn,
    resources_per_trial={"cpu": 4, "gpu": 1},
    config=config,
    num_samples=10,
    scheduler=scheduler,
    progress_reporter=reporter,
    local_dir=output_dir,
    name="cnn_hyperparameter_tuning"
)

# Get the best trial
best_trial = result.get_best_trial("val_loss", "min", "last")
print(f"Best trial config: {best_trial.config}")
print(f"Best trial final validation loss: {best_trial.last_result['val_loss']}")
print(f"Best trial final validation Jaccard score: {best_trial.last_result['val_jaccard']}")

# Load the best model
best_checkpoint_dir = best_trial.checkpoint.value
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_attributes)
model = model.to(device)

checkpoint = torch.load(os.path.join(best_checkpoint_dir, "checkpoint.pt"))
model.load_state_dict(checkpoint["model_state_dict"])

# Save the best model
torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
print(f"Best model saved to {os.path.join(output_dir, 'best_model.pth')}")

# Shutdown Ray
ray.shutdown()

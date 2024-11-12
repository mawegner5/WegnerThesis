import os
import random
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
from ray.air import session
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.concurrency_limiter import ConcurrencyLimiter

# ----------------------------
# Paths and Hyperparameters
# ----------------------------
# Paths
data_dir = '/root/.ipython/WegnerThesis/animals_with_attributes/animals_w_att_data/'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validate')
classes_txt_path = os.path.join(data_dir, 'Animals_with_Attributes2', 'classes.txt')
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
    def __init__(self, root_dir, classes_txt_path, predicates_txt_path, attributes_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Load class names and indices
        self.class_to_idx = {}
        with open(classes_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    idx = int(parts[0]) - 1  # Indices start from 1 in classes.txt
                    cls_name = parts[1].replace('+', ' ')
                    self.class_to_idx[cls_name] = idx

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
        class_names = os.listdir(self.root_dir)
        for cls_name in class_names:
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            # Get class index
            if cls_name not in self.class_to_idx:
                print(f"Class name '{cls_name}' not found in classes.txt")
                continue
            class_idx = self.class_to_idx[cls_name]
            label = self.attribute_matrix[class_idx]
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Use a blank image in case of error
            image = Image.new('RGB', (224, 224))
        label = self.labels[idx]
        label = torch.from_numpy(label).float()
        if self.transform:
            image = self.transform(image)
        return image, label

# ----------------------------
# Data Preparation
# ----------------------------
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
print("Preparing datasets...")
train_dataset = AWA2Dataset(root_dir=train_dir,
                            classes_txt_path=classes_txt_path,
                            predicates_txt_path=predicates_txt_path,
                            attributes_path=attributes_path,
                            transform=data_transform)

val_dataset = AWA2Dataset(root_dir=val_dir,
                          classes_txt_path=classes_txt_path,
                          predicates_txt_path=predicates_txt_path,
                          attributes_path=attributes_path,
                          transform=data_transform)

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")

# ----------------------------
# Hyperparameter Tuning with Ray Tune
# ----------------------------
def train_cnn(config):
    # Create data loaders with num_workers=0
    train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0)

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
    checkpoint = session.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            state = torch.load(checkpoint_path)
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            epoch_start = state["epoch"] + 1
            best_val_loss = state["best_val_loss"]
            trigger_times = state["trigger_times"]
    else:
        epoch_start = 1
        best_val_loss = float('inf')
        trigger_times = 0

    for epoch in range(epoch_start, num_epochs + 1):
        print(f"\nEpoch [{epoch}/{num_epochs}]")
        # Training Phase
        model.train()
        running_loss = 0.0
        all_outputs = []
        all_targets = []

        for images, labels in tqdm(train_loader, desc="Training", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Collect outputs and targets for metric calculation
            outputs = torch.sigmoid(outputs).detach().cpu()
            labels = labels.cpu()
            all_outputs.append(outputs)
            all_targets.append(labels)

        epoch_loss = running_loss / len(train_dataset)

        # Calculate Jaccard Score for training
        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)
        binarized_outputs = (all_outputs > 0.5).float()
        train_jaccard = jaccard_score(all_targets.numpy(), binarized_outputs.numpy(), average='samples')

        print(f"Training Loss: {epoch_loss:.4f}, Training Jaccard Score: {train_jaccard:.4f}")

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_all_outputs = []
        val_all_targets = []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation", leave=False):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)

                outputs = torch.sigmoid(outputs).cpu()
                labels = labels.cpu()
                val_all_outputs.append(outputs)
                val_all_targets.append(labels)

        val_epoch_loss = val_running_loss / len(val_dataset)

        # Calculate Jaccard Score for validation
        val_all_outputs = torch.cat(val_all_outputs)
        val_all_targets = torch.cat(val_all_targets)
        val_binarized_outputs = (val_all_outputs > 0.5).float()
        val_jaccard = jaccard_score(val_all_targets.numpy(), val_binarized_outputs.numpy(), average='samples')

        print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Jaccard Score: {val_jaccard:.4f}")

        # Update scheduler
        scheduler.step()

        # Early Stopping
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            trigger_times = 0
            # Save checkpoint
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "trigger_times": trigger_times
                }, path)
            print("Validation loss decreased. Saving model...")
        else:
            trigger_times += 1
            print(f"EarlyStopping counter: {trigger_times} out of {patience}")
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break

        # Send metrics to Tune
        session.report({
            "val_loss": val_epoch_loss,
            "val_jaccard": val_jaccard,
            "train_loss": epoch_loss,
            "train_jaccard": train_jaccard
        })

# Adjust per-trial resources to use all available resources
train_cnn_with_resources = tune.with_resources(train_cnn, {"cpu": 32, "gpu": 1})

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
    max_t=num_epochs,
    grace_period=10,
    reduction_factor=2
)

reporter = CLIReporter(
    metric_columns=["val_loss", "val_jaccard", "train_loss", "train_jaccard", "training_iteration"]
)

# Limit concurrency to one trial at a time
search_alg = ConcurrencyLimiter(
    BasicVariantGenerator(),
    max_concurrent=1
)

# ----------------------------
# Run Hyperparameter Tuning
# ----------------------------
if __name__ == '__main__':
    # Initialize Ray
    ray.init()

    print("Starting hyperparameter tuning with Ray Tune...")
    tuner = tune.Tuner(
        train_cnn_with_resources,
        param_space=config,
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=scheduler,
            num_samples=10,
            search_alg=search_alg,  # Add the concurrency limiter
        ),
        run_config=ray.air.RunConfig(
            name="cnn_hyperparameter_tuning",
            local_dir=output_dir,
            progress_reporter=reporter,
        )
    )

    results = tuner.fit()

    # Get the best result
    best_result = results.get_best_result(metric="val_loss", mode="min")
    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final validation loss: {best_result.metrics['val_loss']}")
    print(f"Best trial final validation Jaccard score: {best_result.metrics['val_jaccard']}")

    # Load the best model
    best_checkpoint = best_result.checkpoint
    with best_checkpoint.as_directory() as checkpoint_dir:
        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_attributes)
        model = model.to(device)

        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Save the best model
    torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
    print(f"Best model saved to {os.path.join(output_dir, 'best_model.pth')}")

    # Shutdown Ray
    ray.shutdown()

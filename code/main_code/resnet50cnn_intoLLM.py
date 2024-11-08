import os
import torch
import torch.nn as nn  # Import nn module
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import csv
from tqdm import tqdm
import openai  # For OpenAI API calls

# ----------------------------
# Configuration and Hyperparameters
# ----------------------------
# Paths
data_dir = '/root/.ipython/WegnerThesis/animals_with_attributes/animals_w_att_data/'
images_dir = os.path.join(data_dir, 'Animals_with_Attributes2', 'JPEGImages')
test_dir = os.path.join(data_dir, 'test')  # Update if your test images are in a different directory
classes_txt_path = os.path.join(data_dir, 'Animals_with_Attributes2', 'classes.txt')
predicates_txt_path = os.path.join(data_dir, 'Animals_with_Attributes2', 'predicates.txt')
attributes_path = os.path.join(data_dir, 'Animals_with_Attributes2', 'predicate-matrix-binary.txt')
model_path = '/root/.ipython/WegnerThesis/charts_figures_etc/best_model.pth'
output_dir = '/root/.ipython/WegnerThesis/charts_figures_etc/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Hyperparameters
batch_size = 32
num_attributes = 85  # Number of attributes/predicates in AWA2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transforms
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalize with ImageNet mean and std
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# OpenAI API Key
openai.api_key = 'YOUR_OPENAI_API_KEY'  # Replace with your OpenAI API key

# ----------------------------
# Custom Dataset Class
# ----------------------------
class AWA2TestDataset(Dataset):
    def __init__(self, root_dir, classes_txt_path, predicates_txt_path, attributes_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Load class names
        self.classes = []
        self.class_to_idx = {}
        with open(classes_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    idx = int(parts[0]) - 1  # Indices start from 1 in classes.txt
                    cls_name = parts[1]
                    self.classes.append(cls_name)
                    self.class_to_idx[cls_name] = idx
                else:
                    print(f"[Warning] Malformed line in classes.txt: {line.strip()}")

        # Load attribute names
        self.attributes = []
        with open(predicates_txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    idx = int(parts[0]) - 1  # Indices start from 1 in predicates.txt
                    attr_name = parts[1]
                    self.attributes.append(attr_name)
                else:
                    print(f"[Warning] Malformed line in predicates.txt: {line.strip()}")

        # Load attribute matrix
        self.attribute_matrix = np.loadtxt(attributes_path, dtype=int)  # Shape: (N_classes, N_attributes)

        # Build mapping from class index to attribute vector
        self.class_idx_to_attributes = {}
        for idx, cls_name in enumerate(self.classes):
            self.class_idx_to_attributes[idx] = self.attribute_matrix[idx]

        # Collect all image paths and labels
        self.image_paths = []
        self.labels = []
        self.image_classes = []  # Store class names for each image
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue  # Skip if directory does not exist
            class_idx = self.class_to_idx[cls_name]
            label = self.attribute_matrix[class_idx]
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(label)
                    self.image_classes.append(cls_name)
                else:
                    print(f"[Warning] '{img_path}' is not a file, skipping...")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[Error] Could not open image {img_path}: {e}")
            # Return a dummy image or skip
            image = Image.new('RGB', (224, 224))
        label = self.labels[idx]
        label = torch.from_numpy(label).float()  # Convert label to float tensor
        if self.transform:
            image = self.transform(image)
        img_name = os.path.basename(img_path)
        class_name = self.image_classes[idx]
        return image, label, img_name, class_name

# ----------------------------
# Load the Best Model
# ----------------------------
print("Loading the best model...")
from torchvision import models
model = models.resnet50(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_attributes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# ----------------------------
# Prepare Test Dataset and DataLoader
# ----------------------------
print("Preparing test dataset...")
test_dataset = AWA2TestDataset(root_dir=images_dir,  # Assuming test images are in the same directory
                               classes_txt_path=classes_txt_path,
                               predicates_txt_path=predicates_txt_path,
                               attributes_path=attributes_path,
                               transform=data_transform)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# ----------------------------
# Predict Attributes on Test Set
# ----------------------------
print("Predicting attributes on test set...")
attribute_names = test_dataset.attributes  # List of attribute names
test_all_outputs = []
test_image_names = []
test_class_names = []

with torch.no_grad():
    for images, labels, img_names, class_names in tqdm(test_loader, desc="Predicting"):
        images = images.to(device)
        outputs = model(images)
        outputs = torch.sigmoid(outputs).cpu()
        test_all_outputs.extend(outputs)
        test_image_names.extend(img_names)
        test_class_names.extend(class_names)

# ----------------------------
# Feed Predicted Attributes into LLM to Predict Animal Species
# ----------------------------
print("Predicting animal species using LLM...")
predicted_species = []
true_species = []
attribute_descriptions = []

# Prepare prompt template
prompt_template = "Based on the following attributes: {attributes}, predict the animal species. Provide only the species name."

for idx in tqdm(range(len(test_image_names)), desc="LLM Predictions"):
    outputs = test_all_outputs[idx]
    binarized_outputs = (outputs > 0.5).int().numpy()
    predicted_attrs = [attribute_names[i] for i in range(len(attribute_names)) if binarized_outputs[i] == 1]
    attributes_str = ', '.join(predicted_attrs)

    # Prepare prompt
    prompt = prompt_template.format(attributes=attributes_str)

    # Call the OpenAI API
    try:
        response = openai.Completion.create(
            engine='text-davinci-003',  # Or another model you have access to
            prompt=prompt,
            max_tokens=10,
            temperature=0.0,
            n=1,
            stop=None
        )
        llm_output = response.choices[0].text.strip()
        # Keep only the first word (species name), in case of extra text
        species_prediction = llm_output.split('\n')[0].split(',')[0].split('.')[0].strip()
    except Exception as e:
        print(f"[Error] OpenAI API call failed: {e}")
        species_prediction = "Unknown"

    predicted_species.append(species_prediction)
    true_species.append(test_class_names[idx])
    attribute_descriptions.append(attributes_str)

# ----------------------------
# Save Predictions to CSV and Compute Accuracy
# ----------------------------
output_csv_path = os.path.join(output_dir, 'test_predictions.csv')
print(f"Saving predictions to {output_csv_path}...")

with open(output_csv_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['ImageName', 'TrueSpecies', 'PredictedSpecies', 'PredictedAttributes'])
    for idx in range(len(test_image_names)):
        csvwriter.writerow([test_image_names[idx], true_species[idx], predicted_species[idx], attribute_descriptions[idx]])

# Compute accuracy
correct_predictions = sum(1 for true, pred in zip(true_species, predicted_species) if true.lower() == pred.lower())
accuracy = correct_predictions / len(true_species) * 100
print(f"Accuracy on test set: {accuracy:.2f}%")

# ----------------------------
# End of Script
# ----------------------------

import os
import pandas as pd

# Define the file paths
attributes_file = "/root/.ipython/WegnerThesis/data/unzipped_data/CUB_200_2011/attributes.txt"
class_labels_file = "/root/.ipython/WegnerThesis/data/unzipped_data/CUB_200_2011/CUB_200_2011/classes.txt"
class_attribute_file = "/root/.ipython/WegnerThesis/data/unzipped_data/CUB_200_2011/CUB_200_2011/attributes/class_attribute_labels_continuous.txt"

# Load attributes and class labels
with open(attributes_file, 'r') as f:
    attributes = [line.strip().split(' ', 1)[1] for line in f.readlines()]

with open(class_labels_file, 'r') as f:
    class_labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]

# Load class attributes
class_attributes = pd.read_csv(class_attribute_file, delim_whitespace=True, header=None)
class_attributes.columns = attributes
class_attributes.index = class_labels

# Save to Excel or CSV
output_file_xlsx = "/root/.ipython/WegnerThesis/data/unzipped_data/CUB_200_2011/class_attributes.xlsx"
output_file_csv = "/root/.ipython/WegnerThesis/data/unzipped_data/CUB_200_2011/class_attributes.csv"

class_attributes.to_excel(output_file_xlsx)
class_attributes.to_csv(output_file_csv)

print("Data saved to:", output_file_xlsx, "and", output_file_csv)

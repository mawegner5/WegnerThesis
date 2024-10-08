import os

# Define the input and output file paths
input_file_path = '/root/.ipython/WegnerThesis/data/unzipped_data/CUB_200_2011/CUB_200_2011/classes.txt'
output_file_path = '/root/.ipython/WegnerThesis/data/generated_data/cleaned_classes.txt'

# Create a list to hold the cleaned species names
cleaned_species_names = []

# Read the original species names from classes.txt
with open(input_file_path, 'r') as file:
    for line in file:
        # Each line is formatted as "001.Black_footed_Albatross"
        line = line.strip()
        if line:
            # Split the line on the first dot to separate the index from the name
            parts = line.split('.', 1)
            if len(parts) == 2:
                index, species_name = parts
                # Replace underscores with spaces and capitalize each word
                species_name = species_name.replace('_', ' ')
                species_name = species_name.title()  # Capitalize each word
                cleaned_species_names.append(f"{index}.{species_name}")
            else:
                print(f"Unexpected format in line: {line}")

# Save the cleaned species names to the output file
with open(output_file_path, 'w') as file:
    for species in cleaned_species_names:
        file.write(f"{species}\n")

print(f"Cleaned species names saved to: {output_file_path}")

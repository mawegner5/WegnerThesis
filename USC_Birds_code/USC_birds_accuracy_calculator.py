import pandas as pd

# Set the threshold to match the CSV file you want to assess
threshold = 75.0  # Change this to 0.1, 0.2, etc., as needed

# Build the path to the CSV file based on the threshold
csv_file_path = f'/root/.ipython/WegnerThesis/data/generated_data/threshold_{threshold}_LLM_predictions_binned.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Initialize counters for total entries and correct predictions
total = 0
correct = 0

# Function to compare species names, accounting for syntactical differences
def species_match(true_species_name, predicted_species_name):
    # Clean and split species names into sets of words
    true_species_words = set(true_species_name.lower().replace('_', ' ').split())
    predicted_species_words = set(predicted_species_name.lower().replace('_', ' ').split())

    # Check if the predicted species words are a subset of the true species words or vice versa
    if predicted_species_words.issubset(true_species_words):
        return True
    if true_species_words.issubset(predicted_species_words):
        return True
    return False

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    true_species = row['True Species']
    predicted_species_str = row['Predicted Species']

    # Clean the true species name
    if '.' in true_species:
        _, true_species_clean = true_species.split('.', 1)
    else:
        true_species_clean = true_species
    true_species_clean = true_species_clean.replace('_', ' ').strip().lower()

    # Split the predicted species into a list and clean each name
    predicted_species_list = predicted_species_str.split(',')
    predicted_species_clean_list = [species.strip().lower() for species in predicted_species_list]

    # Check for a match between the true species and any of the predicted species
    match_found = False
    for predicted_species in predicted_species_clean_list:
        if species_match(true_species_clean, predicted_species):
            match_found = True
            break

    # Update counters based on whether a match was found
    total += 1
    if match_found:
        correct += 1

# Calculate and print the accuracy
if total > 0:
    accuracy = (correct / total) * 100
    print(f"Accuracy: {accuracy:.2f}%")
else:
    print("No entries to evaluate.")
